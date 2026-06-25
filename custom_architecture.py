import torch
import torch.nn as nn
import transformers
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast
from transformers.models.distilbert.modeling_distilbert import TransformerBlock

# =======================================DEBUGGING====================
import inspect 

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
# print(inspect.signature(TransformerBlock.forward))
print(inspect.signature(TransformerBlock.forward))

# ===============================DEBUGGING ENDS ====================

class GatedRecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Shared transformer block that runs twice (recurrent pass)
        self.shared_layer = TransformerBlock(config)
        # Gate input dimension is hidden_size, output is 1 scalar per token
        # Gate receives pass_a output (not raw hidden_states) for a richer signal
        self.gate = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        # Bias toward -2.0 so gate starts near 0 (favors pass_a output initially)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        # --- First pass through shared layer ---
        # FIX 7: all arguments passed as kwargs — positional order of
        # TransformerBlock.forward() has shifted across transformers versions.
        pass_a_outputs = self.shared_layer(
            hidden_states,
            attn_mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        pass_a = pass_a_outputs[0] if isinstance(pass_a_outputs, tuple) else pass_a_outputs
        if isinstance(pass_a, tuple): 
            pass_a = pass_a[0]# (batch, seq_len, hidden)

        # --- Second (recurrent) pass through the same shared layer ---
        # NOTE: weight sharing deliberately causes distribution shift (each
        # pretrained DistilBERT layer expected unique predecessor outputs).
        # This is an inherent trade-off of this compression strategy, identical
        # in nature to ALBERT's cross-layer sharing. Fine-tuning is expected
        # to re-adapt the shared weights to the recurrent input distribution.
        # Compute cost: 6 effective forward passes — FLOPs reduction is ~0;
        # only parameter count is halved.
        pass_b_outputs = self.shared_layer(
            pass_a,
            attn_mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        pass_b = pass_b_outputs[0] if isinstance(pass_b_outputs, tuple) else pass_b_outputs
        if isinstance(pass_b, tuple): 
            pass_b = pass_b[0]                      # (batch, seq_len, hidden)

        # --- Gating: now conditioned on pass_a (not raw input) ---
        # pass_a carries refined information the gate uses to decide
        # how much of pass_b to blend in.
        # gate_weights shape: (batch, seq_len, 1) — broadcasts over hidden dim
        gate_weights = self.sigmoid(self.gate(pass_a))   # FIX: was self.gate(hidden_states)
        mixed_output = gate_weights * pass_b + (1.0 - gate_weights) * pass_a

        # Preserve extra outputs (e.g. attention weights) from pass_b so they
        # correspond to the same forward pass that produced mixed_output.
        # FIX 2: was returning pass_a_outputs[1:] — those attention maps were
        # computed over hidden_states→pass_a and are mismatched with mixed_output
        # which is derived from pass_b.
        if isinstance(pass_b_outputs, tuple):
            return (mixed_output,) + pass_b_outputs[1:]
        return (mixed_output,)
        # return (mixed_output,) + pass_b_outputs[1:]
        # return (mixed_output,)


class CustomDistilBertModel(DistilBertModel):
    def __init__(self, config: DistilBertConfig):
        super().__init__(config)
        # Replace the 6-layer stack with 3 GatedRecurrentBlocks.
        # Each block runs its shared TransformerBlock twice, giving
        # 6 effective forward passes at ~50 % of the parameter cost.
        self.transformer.layer = nn.ModuleList(
            [GatedRecurrentBlock(config) for _ in range(3)]
        )
        self.config.n_layers = 3
        # post_init() calls _init_weights() which re-initialises every
        # nn.Linear — including each block's gate — with the HuggingFace
        # default (normal, std=0.02, zero bias).  We call it first so our
        # intentional gate bias (-2.0) is applied last and takes effect.
        # FIX 3: post_init() was called before re-applying gate inits,
        # silently overwriting the custom bias that favours pass_a early
        # in training.
        self.post_init()
        self._reinit_gate_biases()

    def _reinit_gate_biases(self) -> None:
        """Re-apply the custom gate initialisation after post_init()."""
        for block in self.transformer.layer:
            nn.init.constant_(block.gate.bias, -2.0)
            nn.init.normal_(block.gate.weight, std=0.02)


# ---------------------------------------------------------------------------
# Weight-transfer helpers
# ---------------------------------------------------------------------------

def _merge_state_dicts(sd_lo: dict, sd_hi: dict) -> dict:
    """
    Merge two consecutive teacher-layer state-dicts into one initialisation
    for a shared student block.

    Strategy per parameter type
    ---------------------------
    LayerNorm (weight / bias)
        These are learned per-layer scale/shift tied tightly to that layer's
        specific output distribution.  Averaging two layers' LayerNorm params
        produces values that are correct for neither distribution and can push
        activations out of the expected range from the very first forward pass.
        FIX 4: LayerNorm params are taken from the *lower* (earlier) teacher
        layer because earlier layers produce more general, transferable features
        and the student block will be used as the first of two recurrent passes.

    All other params (attention projections, FFN weights/biases)
        Element-wise averaging is reasonable — both layers encode similar
        linguistic operations and the average initialises the student closer
        to the true optimum than either extreme alone.
    """
    merged = {}
    for k in sd_lo:
        is_layer_norm = ("layer_norm" in k.lower() or "layernorm" in k.lower())
        if is_layer_norm:
            # Take from lower teacher layer — safer distribution anchor
            merged[k] = sd_lo[k].clone().float()
        else:
            merged[k] = (sd_lo[k].float() + sd_hi[k].float()) / 2.0
    return merged


def build_and_load_poc_model(local_model_path: str) -> CustomDistilBertModel:
    """
    1. Load the pretrained 6-layer DistilBERT as teacher.
    2. Build the 3-block custom model with the same config.
    3. Transfer embeddings verbatim.
    4. Initialise each GatedRecurrentBlock.shared_layer by merging
       consecutive teacher-layer pairs (0+1, 2+3, 4+5):
         - LayerNorm params  → taken from the lower layer (FIX 4)
         - All other params  → element-wise average of both layers
    """
    # --- Load teacher ---
    teacher = DistilBertModel.from_pretrained(
        local_model_path, local_files_only=True
    )
    teacher_layers = teacher.transformer.layer   # 6 TransformerBlocks

    # --- Build student ---
    student = CustomDistilBertModel(teacher.config)

    # --- Transfer embeddings (unchanged) ---
    student.embeddings.load_state_dict(teacher.embeddings.state_dict())

    # --- Transfer transformer weights (selective merge of paired layers) ---
    layer_pairs = [(0, 1), (2, 3), (4, 5)]
    for student_idx, (t_lo, t_hi) in enumerate(layer_pairs):
        merged_sd = _merge_state_dicts(
            teacher_layers[t_lo].state_dict(),
            teacher_layers[t_hi].state_dict(),
        )
        student.transformer.layer[student_idx].shared_layer.load_state_dict(
            merged_sd
        )

    return student


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build and load the custom model
    model = build_and_load_poc_model("./local_distilbert")
    model.to(device)
    model.eval()

    # Load tokenizer for proper mask generation
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "./local_distilbert", local_files_only=True
    )

    # Dummy batch — two sequences of different lengths to exercise masking
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world.",
    ]
    encoded = tokenizer(
        dummy_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    input_ids      = encoded["input_ids"].to(device)       # (B, seq_len)
    attention_mask = encoded["attention_mask"].to(device)  # (B, seq_len)  {0,1}

    # FIX 1: Use the model's own get_extended_attention_mask() instead of
    # manually expanding to (B,1,1,S).  The base-class method handles the
    # {0,1} → additive-float conversion correctly and is resilient to any
    # internal HuggingFace changes to the expected mask format.
    with torch.no_grad():
        # ext_mask = model.get_extended_attention_mask(
        #     attention_mask,
        #     input_ids.shape,
        # )
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # print("Output shape          :", outputs.last_hidden_state.shape)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters      : {total_params:,}")
    print(f"Trainable parameters  : {trainable_params:,}")
    print("config.n_layers =", model.config.n_layers)
    print("actual layers   =", len(model.transformer.layer))


if __name__ == "__main__":
    main()
