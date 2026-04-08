"""
STEP 1: Data Preprocessing Pipeline
=====================================
Project: BERT Fine-Tuning for Twitter Multiclass Text Classification using LoRA on THOS
Dataset : THOS (Targeted Hate and Offensive Speech) + optional Sentiment140

This script:
  - Loads the THOS dataset from HuggingFace
  - Cleans and normalises tweet text
  - Handles class imbalance statistics
  - Saves processed splits (train/val/test) to disk
"""

import re
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from collections import Counter
from sklearn.model_selection import train_test_split

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15
OUTPUT_DIR    = "data/processed"
RAW_DIR       = "data/raw"

# Label mapping  (adjust if THOS uses different strings)
LABEL2ID = {"normal": 0, "offensive": 1, "hate": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR,    exist_ok=True)

# ── Tweet Cleaning ─────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    """
    Normalise a raw tweet for BERT tokenisation.

    Order matters:
      1. Lowercase
      2. Replace URLs with a special token so the model sees context
      3. Replace @mentions
      4. Expand common hashtag compounds lightly (keep the word, remove #)
      5. Remove residual HTML entities
      6. Collapse repeated whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # URLs  →  [URL]
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)

    # @mentions  →  [USER]
    text = re.sub(r"@\w+", "[USER]", text)

    # Hashtags: keep the word, remove the #
    text = re.sub(r"#(\w+)", r"\1", text)

    # HTML entities
    text = re.sub(r"&amp;",  "&",  text)
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&quot;", '"',  text)

    # Remove non-ASCII (emoji, etc.) — comment out to keep emoji
    text = text.encode("ascii", "ignore").decode()

    # Collapse multiple spaces / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Load THOS Dataset ──────────────────────────────────────────────────────────
def load_thos() -> pd.DataFrame:
    """
    Load THOS from HuggingFace hub: mohaimeed/THOS
    Falls back to a local CSV if the hub is unavailable.
    """
    try:
        print("[INFO] Loading THOS from HuggingFace hub …")
        ds = load_dataset("mohaimeed/THOS", trust_remote_code=True)
        # Most HF datasets expose a 'train' split; THOS may only have one split
        split_name = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split_name].to_pandas()
        print(f"[INFO] Loaded {len(df):,} rows from HF hub (split='{split_name}')")
    except Exception as e:
        print(f"[WARN] HF hub failed ({e}). Trying local CSV at {RAW_DIR}/thos.csv …")
        csv_path = os.path.join(RAW_DIR, "thos.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                "Could not load THOS. Either:\n"
                "  (a) run with internet access so HuggingFace hub works, or\n"
                f" (b) place thos.csv in {RAW_DIR}/"
            )
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df):,} rows from local CSV")

    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to canonical 'text' and 'label'.
    Adjust the mapping below to match the actual THOS column names.
    """
    # Common column name variants in THOS
    text_candidates  = ["tweet", "text", "content", "Tweet"]
    label_candidates = ["label", "class", "Category", "category", "Label"]

    text_col  = next((c for c in text_candidates  if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        print("[DEBUG] Available columns:", df.columns.tolist())
        raise ValueError(
            "Could not detect text/label columns automatically. "
            "Please set text_col and label_col manually."
        )

    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df[["text", "label"]]


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string labels → integer ids using LABEL2ID."""
    # Normalise label strings
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL2ID)
    )
    before = len(df)
    df = df.dropna(subset=["label"])          # drop rows with unknown labels
    df["label"] = df["label"].astype(int)
    after = len(df)
    if before != after:
        print(f"[WARN] Dropped {before - after} rows with unrecognised labels")
    return df


# ── Class Imbalance Report ─────────────────────────────────────────────────────
def class_imbalance_report(df: pd.DataFrame, split_name: str = "full") -> None:
    counts = Counter(df["label"])
    total  = len(df)
    print(f"\n{'─'*40}")
    print(f"Class distribution  [{split_name}]  (n={total:,})")
    print(f"{'─'*40}")
    for lid, name in ID2LABEL.items():
        n   = counts.get(lid, 0)
        pct = 100 * n / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:>10}  ({lid})  {n:5,}  {pct:5.1f}%  {bar}")
    print(f"{'─'*40}\n")


# ── Split & Save ───────────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame) -> None:
    """Stratified train / val / test split → CSV files."""
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        split.to_csv(path, index=False)
        print(f"[INFO] Saved {name} split → {path}  ({len(split):,} rows)")
        class_imbalance_report(split, name)


# ── Compute Class Weights for Focal Loss / CrossEntropy ───────────────────────
def compute_class_weights(df: pd.DataFrame) -> dict:
    """
    Inverse-frequency class weights.
    Pass these to torch.nn.CrossEntropyLoss(weight=...) or FocalLoss.
    """
    counts = np.array([Counter(df["label"]).get(i, 1) for i in range(len(LABEL2ID))])
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(LABEL2ID)   # normalise
    weight_dict = {ID2LABEL[i]: round(float(w), 4) for i, w in enumerate(weights)}
    print("[INFO] Computed class weights:", weight_dict)
    return weight_dict


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  STEP 1 — Data Preprocessing")
    print("=" * 55)

    # 1. Load
    df = load_thos()

    # 2. Standardise columns
    df = standardise_columns(df)

    # 3. Clean text
    print("[INFO] Cleaning tweet text …")
    df["text"] = df["text"].apply(clean_tweet)

    # 4. Drop empty rows after cleaning
    df = df[df["text"].str.len() > 0].copy()

    # 5. Encode labels
    df = encode_labels(df)

    # 6. Full-dataset report
    class_imbalance_report(df, "full")

    # 7. Split & save
    split_and_save(df)

    # 8. Compute & save class weights
    train_df     = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
    class_weights = compute_class_weights(train_df)
    import json
    with open(os.path.join(OUTPUT_DIR, "class_weights.json"), "w") as f:
        json.dump(class_weights, f, indent=2)

    print("\n[DONE] Step 1 complete. Files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
