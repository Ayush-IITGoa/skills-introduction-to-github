"""
Fine-tune the custom 3-block gated-recurrent DistilBERT on SST-2 and report
dev accuracy. Compares param count against the standard 6-layer baseline.

Layout assumed (relative to this script):
    ./local_distilbert/                      # pretrained weights + tokenizer
    ./TestFolder/glue_data/sst2/train.tsv    # cols: sentence \t label  (+header)
    ./TestFolder/glue_data/sst2/dev.tsv      # cols: sentence \t label  (+header)
    ./custom_architecture.py                 # your file (importable)

Run:
    python finetune_sst2.py
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

# Import the custom block + merge helpers from your architecture file.
# Rename the import if your local file isn't named custom_architecture.py
from custom_architecture import GatedRecurrentBlock, _merge_state_dicts

LOCAL_MODEL = "./local_distilbert"
SST2_DIR    = "./TestFolder/glue_data/sst2"
MAX_LEN     = 128
BATCH       = 32
EPOCHS      = 3
LR          = 3e-5
WARMUP_FRAC = 0.06
SEED        = 42

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Model: classification head + custom gated-recurrent transformer stack
# ---------------------------------------------------------------------------
class CustomDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config: DistilBertConfig):
        super().__init__(config)
        # Swap the 6-layer stack for 3 GatedRecurrentBlocks (6 effective passes)
        self.distilbert.transformer.layer = nn.ModuleList(
            [GatedRecurrentBlock(config) for _ in range(3)]
        )
        self.post_init()
        # Re-apply intentional gate init AFTER post_init overwrites it.
        # NOTE: bias set to +2.0 so the recurrent pass (pass_b) is actually
        # used from the start — the -2.0 in your PoC discards it at init.
        for block in self.distilbert.transformer.layer:
            nn.init.constant_(block.gate.bias, 2.0)
            nn.init.normal_(block.gate.weight, std=0.02)


def build_student(local_path: str) -> CustomDistilBertForSequenceClassification:
    # Teacher provides config + pretrained weights to seed from.
    teacher = DistilBertForSequenceClassification.from_pretrained(
        local_path, num_labels=2, local_files_only=True
    )
    teacher_layers = teacher.distilbert.transformer.layer

    config = teacher.config
    config.num_labels = 2
    student = CustomDistilBertForSequenceClassification(config)

    # Transfer embeddings verbatim
    student.distilbert.embeddings.load_state_dict(
        teacher.distilbert.embeddings.state_dict()
    )

    # Seed each shared layer from a SINGLE teacher layer (1, 3, 5) rather than
    # averaging pairs — averaging two distinct attention patterns destroys both.
    # Using the deeper layer of each pair tends to transfer better.
    source_layers = [1, 3, 5]
    for s_idx, t_idx in enumerate(source_layers):
        student.distilbert.transformer.layer[s_idx].shared_layer.load_state_dict(
            teacher_layers[t_idx].state_dict()
        )

    return student, teacher


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class SST2Dataset(Dataset):
    def __init__(self, tsv_path, tokenizer):
        self.samples = []
        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader)  # skip header: sentence \t label
            for row in reader:
                if len(row) < 2:
                    continue
                self.samples.append((row[0].strip(), int(row[1])))
        self.tok = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        text, label = self.samples[i]
        enc = self.tok(
            text, truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)   # raw {0,1}; model extends it
            lab  = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds = logits.argmax(dim=-1)
            correct += (preds == lab).sum().item()
            total   += lab.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        LOCAL_MODEL, local_files_only=True
    )

    train_ds = SST2Dataset(os.path.join(SST2_DIR, "train.tsv"), tokenizer)
    dev_ds   = SST2Dataset(os.path.join(SST2_DIR, "dev.tsv"),   tokenizer)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH)

    model, teacher = build_student(LOCAL_MODEL)
    model.to(device)

    # Param comparison
    student_params = sum(p.numel() for p in model.parameters())
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher params : {teacher_params:,}")
    print(f"Student params : {student_params:,}  "
          f"({100*student_params/teacher_params:.1f}% of teacher)")
    del teacher

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_FRAC * total_steps), total_steps
    )

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running += out.loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running/(step+1):.4f}")

        acc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch}: dev accuracy = {acc:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
