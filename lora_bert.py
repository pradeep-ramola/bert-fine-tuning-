"""
STEP 3: BERT + LoRA Fine-Tuning (PEFT)
=======================================
Project: BERT Fine-Tuning for Twitter Multiclass Text Classification using LoRA on THOS

What is LoRA?
  Low-Rank Adaptation (LoRA) freezes the pre-trained BERT weights and injects
  small trainable rank-decomposition matrices (A, B) into the attention layers:
      W_new = W_pretrained + B·A    where rank(B·A) << rank(W)
  This reduces trainable parameters by ~97 % while matching or exceeding full
  fine-tuning accuracy, and prevents catastrophic forgetting on small datasets.

This script:
  - Loads bert-base-uncased and wraps it with PEFT LoraConfig
  - Applies LoRA to q_proj and v_proj (query and value attention matrices)
  - Trains with weighted CrossEntropyLoss (class imbalance fix)
  - Compares # trainable params vs baseline
  - Evaluates and saves best checkpoint

Dependencies:
  pip install peft transformers torch scikit-learn matplotlib seaborn datasets
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
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":        "data/processed",
    "output_dir":      "outputs/lora_bert",
    "model_name":      "bert-base-uncased",
    "max_length":      128,
    "batch_size":      32,
    "num_epochs":      5,
    "learning_rate":   3e-4,      # LoRA typically uses a higher LR than full FT
    "warmup_ratio":    0.1,
    "weight_decay":    0.01,
    "random_seed":     42,
    "num_labels":      3,
    # LoRA hyperparameters
    "lora_r":          8,          # rank — controls capacity vs efficiency trade-off
    "lora_alpha":      16,         # scaling factor (alpha / r = effective LR scale)
    "lora_dropout":    0.1,
    "lora_target_modules": ["query", "value"],   # BERT attention projections
}

LABEL2ID = {"normal": 0, "offensive": 1, "hate": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")



class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts   = df["text"].tolist()
        self.labels  = df["label"].tolist()
        self.tok     = tokenizer
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
            "token_type_ids": enc.get("token_type_ids",
                               torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }



def count_parameters(model) -> tuple[int, int]:
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def print_parameter_report(model, label: str = "LoRA BERT") -> None:
    trainable, total = count_parameters(model)
    print(f"\n[PARAMS] {label}")
    print(f"  Trainable : {trainable:>12,}  ({100*trainable/total:.2f} % of total)")
    print(f"  Frozen    : {total-trainable:>12,}")
    print(f"  Total     : {total:>12,}\n")


# ── Build LoRA Model ──────────────────────────────────────────────────────────
def build_lora_model() -> BertForSequenceClassification:
    """
    Wrap BertForSequenceClassification with LoRA adapters.
    Only the LoRA matrices (A, B) and the classifier head are trainable.
    """
    base_model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(base_model, lora_config)
    return model


#Training Loop 
def train_epoch(model, loader, optimizer, scheduler, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), macro_f1, all_preds, all_labels


#Plots 
def plot_confusion_matrix(labels, preds, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=list(ID2LABEL.values()),
        yticklabels=list(ID2LABEL.values()),
        ax=ax,
    )
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved → {save_path}")


def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("LoRA — Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")
    axes[1].plot(history["val_f1"], label="Val Macro F1", color="green")
    axes[1].set_title("LoRA — Val Macro F1"); axes[1].legend(); axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] History plot saved → {save_path}")



def main():
    print("=" * 55)
    print("  STEP 3 — BERT + LoRA Fine-Tuning")
    print("=" * 55)

    # 1. Load data
    train_df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    val_df   = pd.read_csv(os.path.join(CONFIG["data_dir"], "val.csv"))
    test_df  = pd.read_csv(os.path.join(CONFIG["data_dir"], "test.csv"))
    print(f"[INFO] train:{len(train_df):,}  val:{len(val_df):,}  test:{len(test_df):,}")

    # 2. Tokeniser
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["model_name"])

    # 3. Datasets & loaders
    train_loader = DataLoader(TweetDataset(train_df, tokenizer, CONFIG["max_length"]),
                              batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
    val_loader   = DataLoader(TweetDataset(val_df, tokenizer, CONFIG["max_length"]),
                              batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(TweetDataset(test_df, tokenizer, CONFIG["max_length"]),
                              batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

    # 4. Build LoRA model
    model = build_lora_model()
    model.to(DEVICE)
    print_parameter_report(model)
    model.print_trainable_parameters()   # PEFT built-in summary

    # 5. Loss (weighted)
    with open(os.path.join(CONFIG["data_dir"], "class_weights.json")) as f:
        w = json.load(f)
    class_weights = torch.tensor(
        [w[ID2LABEL[i]] for i in range(CONFIG["num_labels"])], dtype=torch.float
    ).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 6. Optimizer — only optimise trainable (LoRA) params
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    total_steps  = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 7. Training loop
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
            # Save PEFT adapter weights only (tiny file!)
            model.save_pretrained(best_ckpt)
            tokenizer.save_pretrained(best_ckpt)
            print(f"    ✓ New best LoRA adapter saved (val F1={val_f1:.4f})")

    # 8. Test evaluation
    print("\n[INFO] Loading best LoRA checkpoint for test evaluation …")
    base_model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    best_model = PeftModel.from_pretrained(base_model, best_ckpt)
    best_model.to(DEVICE)

    _, test_f1, test_preds, test_labels = evaluate(best_model, test_loader, loss_fn)
    print(f"\n[RESULT] LoRA Test Macro F1: {test_f1:.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=list(ID2LABEL.values())))

    # 9. Plots
    plot_history(history, os.path.join(CONFIG["output_dir"], "training_history.png"))
    plot_confusion_matrix(
        test_labels, test_preds,
        os.path.join(CONFIG["output_dir"], "confusion_matrix_test.png"),
        title="LoRA BERT — Test Confusion Matrix"
    )

    # 10. Save results
    results = {
        "model":          "lora_bert",
        "lora_r":         CONFIG["lora_r"],
        "lora_alpha":     CONFIG["lora_alpha"],
        "test_macro_f1":  round(test_f1, 4),
        "best_val_f1":    round(best_val_f1, 4),
        "config":         CONFIG,
    }
    with open(os.path.join(CONFIG["output_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Step 3 complete. Outputs in: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
