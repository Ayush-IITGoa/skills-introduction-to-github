"""
Multi-task GLUE runner for the custom 3-block gated-recurrent DistilBERT.

Tasks (chosen for breadth):
    MRPC  - sentence-pair paraphrase, small        -> accuracy + F1
    STS-B - sentence-pair similarity, REGRESSION    -> Pearson + Spearman
    QNLI  - question/sentence entailment, mid-size  -> accuracy
    RTE   - textual entailment, tiny (stress test)  -> accuracy

Layout assumed (relative to this script):
    ./local_distilbert/
    ./TestFolder/glue_data/mrpc/{train,dev}.tsv
    ./TestFolder/glue_data/stsb/{train,dev}.tsv
    ./TestFolder/glue_data/qnli/{train,dev}.tsv
    ./TestFolder/glue_data/rte/{train,dev}.tsv
    ./custom_architecture.py

Run:
    python glue_runner.py              # all four
    python glue_runner.py mrpc rte     # subset
"""

import os
import sys
import csv
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from custom_architecture import GatedRecurrentBlock

LOCAL_MODEL = "./local_distilbert"
GLUE_DIR    = "./TestFolder/glue_data"
MAX_LEN     = 128
SEED        = 42
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Per-task config. Column indices are 0-based into the TSV row.
# These match the canonical GLUE .tsv layouts (the ones from the GLUE
# download script). If your files differ, adjust cols/skip_header here ONLY.
# ---------------------------------------------------------------------------
TASKS = {
    "mrpc": {
        # MRPC train/dev: Quality \t #1 ID \t #2 ID \t #1 String \t #2 String
        "text_a": 3, "text_b": 4, "label": 0,
        "num_labels": 2, "regression": False,
        "epochs": 5, "lr": 3e-5, "batch": 16,
        "metric": "acc_f1",
    },
    "stsb": {
        # STS-B train/dev: ...10 meta cols... sentence1[7] sentence2[8] score[9]
        "text_a": 7, "text_b": 8, "label": 9,
        "num_labels": 1, "regression": True,
        "epochs": 5, "lr": 3e-5, "batch": 16,
        "metric": "pearson_spearman",
    },
    "qnli": {
        # QNLI train/dev: index \t question \t sentence \t label(entailment/not)
        "text_a": 1, "text_b": 2, "label": 3,
        "num_labels": 2, "regression": False,
        "epochs": 3, "lr": 3e-5, "batch": 32,
        "metric": "acc",
        "label_map": {"entailment": 0, "not_entailment": 1},
    },
    "rte": {
        # RTE train/dev: index \t sentence1 \t sentence2 \t label
        "text_a": 1, "text_b": 2, "label": 3,
        "num_labels": 2, "regression": False,
        "epochs": 5, "lr": 2e-5, "batch": 16,
        "metric": "acc",
        "label_map": {"entailment": 0, "not_entailment": 1},
    },
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CustomForSeqClass(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert.transformer.layer = nn.ModuleList(
            [GatedRecurrentBlock(config) for _ in range(3)]
        )
        self.post_init()
        for block in self.distilbert.transformer.layer:
            nn.init.constant_(block.gate.bias, 2.0)   # +2.0: use the recurrent pass
            nn.init.normal_(block.gate.weight, std=0.02)


def build_student(num_labels, problem_type=None):
    teacher = DistilBertForSequenceClassification.from_pretrained(
        LOCAL_MODEL, num_labels=num_labels, local_files_only=True
    )
    config = teacher.config
    config.num_labels = num_labels
    if problem_type:
        config.problem_type = problem_type   # "regression" for STS-B
    student = CustomForSeqClass(config)

    student.distilbert.embeddings.load_state_dict(
        teacher.distilbert.embeddings.state_dict()
    )
    # Seed shared layers from single teacher layers 1,3,5 (no averaging)
    for s_idx, t_idx in enumerate([1, 3, 5]):
        student.distilbert.transformer.layer[s_idx].shared_layer.load_state_dict(
            teacher.distilbert.transformer.layer[t_idx].state_dict()
        )
    s_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    del teacher
    return student, s_params, t_params


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class GlueDataset(Dataset):
    def __init__(self, path, cfg, tokenizer):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)  # header
            for row in reader:
                need = max(cfg["text_a"], cfg["text_b"] or 0, cfg["label"])
                if len(row) <= need:
                    continue
                a = row[cfg["text_a"]].strip()
                b = row[cfg["text_b"]].strip() if cfg["text_b"] is not None else None
                raw = row[cfg["label"]].strip()
                if cfg["regression"]:
                    label = float(raw)
                elif "label_map" in cfg:
                    if raw not in cfg["label_map"]:
                        continue
                    label = cfg["label_map"][raw]
                else:
                    label = int(raw)
                self.rows.append((a, b, label))
        self.cfg = cfg
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, b, label = self.rows[i]
        enc = self.tok(
            a, b, truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        item["labels"] = (torch.tensor(label, dtype=torch.float)
                          if self.cfg["regression"]
                          else torch.tensor(label, dtype=torch.long))
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _f1(preds, labels):
    tp = sum(p == 1 and l == 1 for p, l in zip(preds, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(preds, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(preds, labels))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

def _pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a-mx)**2 for a in x))
    vy = math.sqrt(sum((b-my)**2 for b in y))
    return cov / (vx*vy) if vx and vy else 0.0

def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v)
        for rank_i, idx in enumerate(order):
            r[idx] = rank_i
        return r
    return _pearson(rank(x), rank(y))


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def evaluate(model, loader, cfg, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            if cfg["regression"]:
                preds.extend(logits.squeeze(-1).cpu().tolist())
            else:
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    if cfg["metric"] == "acc":
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        return {"acc": acc}
    if cfg["metric"] == "acc_f1":
        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        return {"acc": acc, "f1": _f1(preds, labels)}
    if cfg["metric"] == "pearson_spearman":
        return {"pearson": _pearson(preds, labels),
                "spearman": _spearman(preds, labels)}


def run_task(name, tokenizer, device):
    cfg = TASKS[name]
    tdir = os.path.join(GLUE_DIR, name)
    print(f"\n{'='*60}\nTASK: {name.upper()}\n{'='*60}")

    problem_type = "regression" if cfg["regression"] else None
    model, s_params, t_params = build_student(cfg["num_labels"], problem_type)
    model.to(device)
    print(f"Student params: {s_params:,} ({100*s_params/t_params:.1f}% of teacher)")

    train_ds = GlueDataset(os.path.join(tdir, "train.tsv"), cfg, tokenizer)
    dev_ds   = GlueDataset(os.path.join(tdir, "dev.tsv"),   cfg, tokenizer)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg["batch"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.06 * total_steps), total_steps
    )

    best = None
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        metrics = evaluate(model, dev_loader, cfg, device)
        primary = list(metrics.values())[0]
        if best is None or primary > list(best.values())[0]:
            best = metrics
        msg = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  epoch {epoch}: {msg}")

    print(f"BEST {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in best.items()))
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        LOCAL_MODEL, local_files_only=True
    )

    requested = [a.lower() for a in sys.argv[1:]] or list(TASKS.keys())
    results = {}
    for name in requested:
        if name not in TASKS:
            print(f"skip unknown task: {name}")
            continue
        results[name] = run_task(name, tokenizer, device)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for name, m in results.items():
        print(f"{name.upper():6s}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
