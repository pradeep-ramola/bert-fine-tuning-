"""
STEP 2: Baseline BERT Fine-Tuning on THOS
==========================================
Project: BERT Fine-Tuning for Twitter Multiclass Text Classification using LoRA on THOS

This script:
  - Loads the preprocessed train/val/test CSVs from Step 1
  - Tokenises with bert-base-uncased
  - Fine-tunes the full BERT model (baseline — no LoRA yet)
  - Evaluates with macro F1, per-class F1, confusion matrix
  - Saves the model checkpoint for comparison with LoRA version

Run AFTER data_preprocessing.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")          # headless (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":        "data/processed",
    "output_dir":      "outputs/baseline_bert",
    "model_name":      "bert-base-uncased",
    "max_length":      128,        # 128 covers ~95 % of tweets
    "batch_size":      32,
    "num_epochs":      5,
    "learning_rate":   2e-5,
    "warmup_ratio":    0.1,
    "weight_decay":    0.01,
    "random_seed":     42,
    "num_labels":      3,
}

LABEL2ID = {"normal": 0, "offensive": 1, "hate": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ── Dataset Class ─────────────────────────────────────────────────────────────
class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Load Data ─────────────────────────────────────────────────────────────────
def load_splits(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"[INFO] Splits loaded — train:{len(train):,}  val:{len(val):,}  test:{len(test):,}")
    return train, val, test


# ── Load Class Weights ─────────────────────────────────────────────────────────
def load_class_weights(data_dir: str) -> torch.Tensor:
    path = os.path.join(data_dir, "class_weights.json")
    with open(path) as f:
        w = json.load(f)
    weights = torch.tensor(
        [w[ID2LABEL[i]] for i in range(CONFIG["num_labels"])],
        dtype=torch.float
    )
    print(f"[INFO] Class weights: {weights.tolist()}")
    return weights.to(DEVICE)


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


#Evaluation 
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), macro_f1, all_preds, all_labels


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=list(ID2LABEL.values()),
        yticklabels=list(ID2LABEL.values()),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved → {save_path}")


#Training History Plot 
def plot_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_f1"], label="Val Macro F1", color="green")
    axes[1].set_title("Validation Macro F1")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training history saved → {save_path}")


#Main
def main():
    print("=" * 55)
    print("  STEP 2 — Baseline BERT Fine-Tuning")
    print("=" * 55)

    # Load data
    train_df, val_df, test_df = load_splits(CONFIG["data_dir"])

    #Tokeniser
    print(f"[INFO] Loading tokeniser: {CONFIG['model_name']}")
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])

    #Datasets & DataLoaders
    train_ds = TweetDataset(train_df, tokenizer, CONFIG["max_length"])
    val_ds   = TweetDataset(val_df,   tokenizer, CONFIG["max_length"])
    test_ds  = TweetDataset(test_df,  tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    #Model
    print(f"[INFO] Loading model: {CONFIG['model_name']}")
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    #Loss (with class weights to handle imbalance)
    class_weights = load_class_weights(CONFIG["data_dir"])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    #Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    total_steps   = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    #Training loop
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_f1 = 0.0
    best_ckpt   = os.path.join(CONFIG["output_dir"], "best_model")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"  Epoch {epoch}/{CONFIG['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"    ✓ New best model saved (val F1={val_f1:.4f})")

    #Final Test Evaluation
    print("\n[INFO] Loading best checkpoint for test evaluation …")
    best_model = BertForSequenceClassification.from_pretrained(best_ckpt)
    best_model.to(DEVICE)

    _, test_f1, test_preds, test_labels = evaluate(best_model, test_loader, loss_fn)
    print(f"\n[RESULT] Test Macro F1: {test_f1:.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=list(ID2LABEL.values())))

    #Plots
    plot_history(
        history,
        os.path.join(CONFIG["output_dir"], "training_history.png")
    )
    plot_confusion_matrix(
        test_labels, test_preds,
        os.path.join(CONFIG["output_dir"], "confusion_matrix_test.png"),
        title="Baseline BERT — Test Confusion Matrix"
    )

    #Save results summary
    results = {
        "model":       "baseline_bert",
        "test_macro_f1": round(test_f1, 4),
        "best_val_f1":   round(best_val_f1, 4),
        "config":        CONFIG,
    }
    with open(os.path.join(CONFIG["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Step 2 complete. Outputs in: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
