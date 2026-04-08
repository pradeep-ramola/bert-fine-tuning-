"""
STEP 4: Class Imbalance — Focal Loss Implementation
====================================================
Project: BERT Fine-Tuning for Twitter Multiclass Text Classification using LoRA on THOS

Why Focal Loss?
  THOS is heavily skewed (Normal >> Offensive >> Hate).
  Standard CrossEntropyLoss treats all samples equally, so the model learns to
  predict the majority class. Focal Loss down-weights easy/correct predictions
  and focuses training on hard, misclassified examples:

      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

  - gamma > 0: reduces loss for easy examples (confident correct predictions)
  - alpha:     per-class weighting (same purpose as class_weight in CE)
  - gamma = 0: reduces to standard weighted CE

This script:
  - Implements a multiclass FocalLoss module
  - Re-trains the LoRA model from Step 3 with FocalLoss
  - Compares per-class F1 vs Step 3 (weighted CE only)
  - This file can also be imported by step3_lora_bert.py for a drop-in swap

Usage:
  Run standalone  →  python step4_focal_loss.py
  Import          →  from step4_focal_loss import FocalLoss
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":        "data/processed",
    "output_dir":      "outputs/lora_focal",
    "model_name":      "bert-base-uncased",
    "max_length":      128,
    "batch_size":      32,
    "num_epochs":      5,
    "learning_rate":   3e-4,
    "warmup_ratio":    0.1,
    "weight_decay":    0.01,
    "random_seed":     42,
    "num_labels":      3,
    "lora_r":          8,
    "lora_alpha":      16,
    "lora_dropout":    0.1,
    "lora_target_modules": ["query", "value"],
    # Focal Loss specific
    "focal_gamma":     2.0,        # focusing parameter (2 is a widely used default)
}

LABEL2ID = {"normal": 0, "offensive": 1, "hate": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss.

    Args:
        gamma  (float): Focusing parameter. 0 = standard CE.
        alpha  (Tensor | None): Per-class weight tensor of shape (num_classes,).
                                If None, all classes are weighted equally.
        reduction (str): 'mean' | 'sum' | 'none'

    Reference: Lin et al. (2017) https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha          # shape (C,) or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (N, C)  raw (unnormalised) class scores
            targets : (N,)    integer class indices
        Returns:
            Scalar loss value
        """
        # Standard CE loss (per-sample, unreduced)
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")

        # p_t  = probability assigned to the correct class
        probs   = F.softmax(logits, dim=-1)                       # (N, C)
        p_t     = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # (N,)

        # Focal weight
        focal_weight = (1.0 - p_t) ** self.gamma                 # (N,)

        # Focal loss
        loss = focal_weight * ce_loss                             # (N,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def __repr__(self):
        return f"FocalLoss(gamma={self.gamma}, alpha={self.alpha})"


# ── Dataset ────────────────────────────────────────────────────────────────────
class TweetDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts   = df["text"].tolist()
        self.labels  = df["label"].tolist()
        self.tok     = tokenizer
        self.max_len = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids",
                               torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Build LoRA Model (same as Step 3) ────────────────────────────────────────
def build_lora_model():
    base = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_labels"],
        id2label=ID2LABEL, label2id=LABEL2ID)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none", inference_mode=False)
    return get_peft_model(base, lora_cfg)


# ── Training & Evaluation ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train(); total = 0
    for batch in loader:
        optimizer.zero_grad()
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        tids = batch["token_type_ids"].to(DEVICE)
        labs = batch["labels"].to(DEVICE)
        logits = model(input_ids=ids, attention_mask=mask, token_type_ids=tids).logits
        loss   = loss_fn(logits, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += loss.item()
    return total / len(loader)


def evaluate(model, loader, loss_fn):
    model.eval(); total = 0; preds_all = []; labs_all = []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            tids = batch["token_type_ids"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=tids).logits
            total += loss_fn(logits, labs).item()
            preds_all.extend(logits.argmax(-1).cpu().numpy())
            labs_all.extend(labs.cpu().numpy())
    return total / len(loader), f1_score(labs_all, preds_all, average="macro"), preds_all, labs_all


# ── Visualisation helpers ─────────────────────────────────────────────────────
def plot_per_class_f1_comparison(report_baseline: dict, report_focal: dict, save_path: str):
    """
    Bar chart comparing per-class F1 from weighted CE (Step 3) vs Focal Loss.
    Expects sklearn classification_report dicts.
    """
    classes = list(ID2LABEL.values())
    f1_base  = [report_baseline.get(c, {}).get("f1-score", 0) for c in classes]
    f1_focal = [report_focal.get(c, {}).get("f1-score", 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, f1_base,  w, label="Weighted CE (LoRA)",  color="steelblue")
    ax.bar(x + w/2, f1_focal, w, label="Focal Loss (LoRA)",   color="coral")
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1: Weighted CE vs Focal Loss")
    ax.legend(); plt.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[INFO] F1 comparison plot → {save_path}")


def plot_confusion_matrix(labels, preds, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=list(ID2LABEL.values()),
                yticklabels=list(ID2LABEL.values()), ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"[INFO] Confusion matrix → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 4 — LoRA + Focal Loss (Class Imbalance Fix)")
    print("=" * 55)

    # 1. Data
    train_df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    val_df   = pd.read_csv(os.path.join(CONFIG["data_dir"], "val.csv"))
    test_df  = pd.read_csv(os.path.join(CONFIG["data_dir"], "test.csv"))
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])

    train_loader = DataLoader(TweetDataset(train_df, tokenizer, CONFIG["max_length"]),
                              batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(TweetDataset(val_df,   tokenizer, CONFIG["max_length"]),
                              batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(TweetDataset(test_df,  tokenizer, CONFIG["max_length"]),
                              batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 2. Focal Loss with class-weighted alpha
    with open(os.path.join(CONFIG["data_dir"], "class_weights.json")) as f:
        w = json.load(f)
    alpha = torch.tensor([w[ID2LABEL[i]] for i in range(CONFIG["num_labels"])],
                         dtype=torch.float).to(DEVICE)
    loss_fn = FocalLoss(gamma=CONFIG["focal_gamma"], alpha=alpha)
    print(f"[INFO] {loss_fn}")

    # 3. LoRA model
    model = build_lora_model()
    model.to(DEVICE)
    model.print_trainable_parameters()

    # 4. Optimizer / scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    total_steps  = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 5. Train
    history  = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1  = 0.0
    best_ckpt = os.path.join(CONFIG["output_dir"], "best_model")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        tl = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        vl, vf1, _, _ = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_f1"].append(vf1)
        print(f"  Epoch {epoch} | Train Loss:{tl:.4f}  Val Loss:{vl:.4f}  Val F1:{vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"    ✓ Saved best model (F1={vf1:.4f})")

    # 6. Test
    base_model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=CONFIG["num_labels"],
        id2label=ID2LABEL, label2id=LABEL2ID)
    best_model = PeftModel.from_pretrained(base_model, best_ckpt).to(DEVICE)
    _, test_f1, test_preds, test_labels = evaluate(best_model, test_loader, loss_fn)

    print(f"\n[RESULT] Focal Loss LoRA — Test Macro F1: {test_f1:.4f}\n")
    report = classification_report(test_labels, test_preds,
                                   target_names=list(ID2LABEL.values()),
                                   output_dict=True)
    print(classification_report(test_labels, test_preds,
                                target_names=list(ID2LABEL.values())))

    # 7. Compare with Step 3 results (if they exist)
    step3_results_path = "outputs/lora_bert/results.json"
    if os.path.exists(step3_results_path):
        with open(step3_results_path) as f:
            step3 = json.load(f)
        print(f"\n[COMPARE] Step 3 (Weighted CE)  Test Macro F1: {step3['test_macro_f1']}")
        print(f"[COMPARE] Step 4 (Focal Loss)    Test Macro F1: {round(test_f1, 4)}")

    # 8. Plots
    plot_confusion_matrix(
        test_labels, test_preds,
        os.path.join(CONFIG["output_dir"], "confusion_matrix_test.png"),
        "Focal Loss LoRA — Test Confusion Matrix"
    )

    # 9. Save results
    results = {
        "model":         "lora_focal_loss",
        "focal_gamma":   CONFIG["focal_gamma"],
        "test_macro_f1": round(test_f1, 4),
        "best_val_f1":   round(best_f1, 4),
        "per_class_f1": {
            cls: round(report[cls]["f1-score"], 4)
            for cls in list(ID2LABEL.values()) if cls in report
        },
        "config": CONFIG,
    }
    with open(os.path.join(CONFIG["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Step 4 complete. Outputs in: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
