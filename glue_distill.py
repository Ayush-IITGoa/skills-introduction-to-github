"""
GLUE runner with KNOWLEDGE DISTILLATION for the custom gated-recurrent DistilBERT.

Per task, two phases:
  Phase 1: fine-tune a full 6-layer DistilBERT teacher on the task.
  Phase 2: distill the 3-block student from that teacher.
           loss = alpha * CE(student, hard_labels)
                + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)
           (STS-B is regression: MSE to teacher logits instead of KL.)

Tasks: mrpc, stsb, qnli, rte  (same breadth set as before)

Layout (relative to this script):
    ./local_distilbert/
    ./TestFolder/glue_data/{mrpc,stsb,qnli,rte}/{train,dev}.tsv
    ./custom_architecture.py

Run:
    python glue_distill.py            # all four
    python glue_distill.py rte mrpc   # subset
"""

import os
import sys
import csv
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

from custom_architecture import GatedRecurrentBlock

LOCAL_MODEL = "./local_distilbert"
GLUE_DIR    = "./TestFolder/glue_data"
MAX_LEN     = 128
SEED        = 42

# Distillation hyperparameters
ALPHA       = 0.5    # weight on hard-label loss; (1-ALPHA) on distillation loss
TEMPERATURE = 4.0    # softmax temperature for soft targets

torch.manual_seed(SEED)

TASKS = {
    "mrpc": {"text_a": 3, "text_b": 4, "label": 0, "num_labels": 2,
             "regression": False, "epochs": 5, "lr": 3e-5, "batch": 16,
             "metric": "acc_f1"},
    "stsb": {"text_a": 7, "text_b": 8, "label": 9, "num_labels": 1,
             "regression": True,  "epochs": 5, "lr": 3e-5, "batch": 16,
             "metric": "pearson_spearman"},
    "qnli": {"text_a": 1, "text_b": 2, "label": 3, "num_labels": 2,
             "regression": False, "epochs": 3, "lr": 3e-5, "batch": 32,
             "metric": "acc", "label_map": {"entailment": 0, "not_entailment": 1}},
    "rte":  {"text_a": 1, "text_b": 2, "label": 3, "num_labels": 2,
             "regression": False, "epochs": 6, "lr": 2e-5, "batch": 16,
             "metric": "acc", "label_map": {"entailment": 0, "not_entailment": 1}},
}


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------
class CustomForSeqClass(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert.transformer.layer = nn.ModuleList(
            [GatedRecurrentBlock(config) for _ in range(3)]
        )
        self.post_init()
        for block in self.distilbert.transformer.layer:
            nn.init.constant_(block.gate.bias, 2.0)
            nn.init.normal_(block.gate.weight, std=0.02)


def build_teacher(num_labels, problem_type):
    teacher = DistilBertForSequenceClassification.from_pretrained(
        LOCAL_MODEL, num_labels=num_labels, local_files_only=True
    )
    if problem_type:
        teacher.config.problem_type = problem_type
    return teacher


def build_student(teacher, num_labels, problem_type):
    config = copy.deepcopy(teacher.config)
    config.num_labels = num_labels
    if problem_type:
        config.problem_type = problem_type
    student = CustomForSeqClass(config)
    student.distilbert.embeddings.load_state_dict(
        teacher.distilbert.embeddings.state_dict()
    )
    for s_idx, t_idx in enumerate([1, 3, 5]):
        student.distilbert.transformer.layer[s_idx].shared_layer.load_state_dict(
            teacher.distilbert.transformer.layer[t_idx].state_dict()
        )
    return student


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class GlueDataset(Dataset):
    def __init__(self, path, cfg, tokenizer):
        self.rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            next(reader)
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
        self.cfg, self.tok = cfg, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        a, b, label = self.rows[i]
        enc = self.tok(a, b, truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        item = {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0)}
        item["labels"] = (torch.tensor(label, dtype=torch.float)
                          if self.cfg["regression"]
                          else torch.tensor(label, dtype=torch.long))
        return item


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _f1(p, l):
    tp = sum(a == 1 and b == 1 for a, b in zip(p, l))
    fp = sum(a == 1 and b == 0 for a, b in zip(p, l))
    fn = sum(a == 0 and b == 1 for a, b in zip(p, l))
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def _pearson(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a-mx)**2 for a in x)); vy = math.sqrt(sum((b-my)**2 for b in y))
    return cov/(vx*vy) if vx and vy else 0.0

def _spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
        for ri, idx in enumerate(order): r[idx] = ri
        return r
    return _pearson(rank(x), rank(y))


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
        return {"acc": sum(p == l for p, l in zip(preds, labels))/len(labels)}
    if cfg["metric"] == "acc_f1":
        return {"acc": sum(p == l for p, l in zip(preds, labels))/len(labels),
                "f1": _f1(preds, labels)}
    return {"pearson": _pearson(preds, labels), "spearman": _spearman(preds, labels)}


# ---------------------------------------------------------------------------
# Phase 1: train teacher
# ---------------------------------------------------------------------------
def train_teacher(teacher, train_loader, dev_loader, cfg, device):
    teacher.to(device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = len(train_loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)
    for epoch in range(cfg["epochs"]):
        teacher.train()
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)
            opt.zero_grad()
            out = teacher(input_ids=ids, attention_mask=mask, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step(); sched.step()
        m = evaluate(teacher, dev_loader, cfg, device)
        print("  [teacher] epoch %d: %s" % (epoch,
              "  ".join(f"{k}={v:.4f}" for k, v in m.items())))
    return teacher


# ---------------------------------------------------------------------------
# Phase 2: distill student from teacher
# ---------------------------------------------------------------------------
def distill_loss(student_logits, teacher_logits, labels, cfg):
    if cfg["regression"]:
        # Regression: student matches teacher's scalar output (MSE) + true MSE
        hard = F.mse_loss(student_logits.squeeze(-1), labels)
        soft = F.mse_loss(student_logits.squeeze(-1), teacher_logits.squeeze(-1))
        return ALPHA * hard + (1 - ALPHA) * soft

    hard = F.cross_entropy(student_logits, labels)
    s_log = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    t_prob = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
    soft = F.kl_div(s_log, t_prob, reduction="batchmean") * (TEMPERATURE ** 2)
    return ALPHA * hard + (1 - ALPHA) * soft


def distill_student(student, teacher, train_loader, dev_loader, cfg, device):
    student.to(device); teacher.to(device); teacher.eval()
    opt = torch.optim.AdamW(student.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps = len(train_loader) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.06*steps), steps)
    best = None
    for epoch in range(cfg["epochs"]):
        student.train()
        for batch in train_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lab  = batch["labels"].to(device)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids, attention_mask=mask).logits
            s_logits = student(input_ids=ids, attention_mask=mask).logits
            loss = distill_loss(s_logits, t_logits, lab, cfg)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step()
        m = evaluate(student, dev_loader, cfg, device)
        if best is None or list(m.values())[0] > list(best.values())[0]:
            best = m
        print("  [student] epoch %d: %s" % (epoch,
              "  ".join(f"{k}={v:.4f}" for k, v in m.items())))
    return best


def run_task(name, tokenizer, device):
    cfg = TASKS[name]
    tdir = os.path.join(GLUE_DIR, name)
    print(f"\n{'='*60}\nTASK: {name.upper()}\n{'='*60}")
    problem_type = "regression" if cfg["regression"] else None

    train_ds = GlueDataset(os.path.join(tdir, "train.tsv"), cfg, tokenizer)
    dev_ds   = GlueDataset(os.path.join(tdir, "dev.tsv"),   cfg, tokenizer)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=cfg["batch"])

    print("Phase 1: fine-tuning teacher")
    teacher = build_teacher(cfg["num_labels"], problem_type)
    teacher = train_teacher(teacher, train_loader, dev_loader, cfg, device)

    print("Phase 2: distilling student")
    student = build_student(teacher, cfg["num_labels"], problem_type)
    s_params = sum(p.numel() for p in student.parameters())
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"Student params: {s_params:,} ({100*s_params/t_params:.1f}% of teacher)")
    best = distill_student(student, teacher, train_loader, dev_loader, cfg, device)

    print(f"BEST {name}: " + "  ".join(f"{k}={v:.4f}" for k, v in best.items()))
    del teacher, student
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| alpha:", ALPHA, "| T:", TEMPERATURE)
    tokenizer = DistilBertTokenizerFast.from_pretrained(LOCAL_MODEL, local_files_only=True)

    requested = [a.lower() for a in sys.argv[1:]] or list(TASKS.keys())
    results = {}
    for name in requested:
        if name not in TASKS:
            print("skip unknown task:", name); continue
        results[name] = run_task(name, tokenizer, device)

    print(f"\n{'='*60}\nSUMMARY (distilled student)\n{'='*60}")
    for name, m in results.items():
        print(f"{name.upper():6s}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
