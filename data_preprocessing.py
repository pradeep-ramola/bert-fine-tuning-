"""
STEP 1: Data Preprocessing Pipeline
=====================================
Project: BERT Fine-Tuning for Twitter Multiclass Text Classification using LoRA on THOS

Dataset source: https://github.com/mohaimeed/THOS
Two JSON files must be placed at:
  data/raw/THOS_Dataset_Text.json   ← tweet texts,  [{index, text}, ...]
  data/raw/THOS_Dataset.json        ← annotations,  [{index, col2, col3, ...}, ...]

THOS label schema (from the paper / GitHub README):
  Column 2 (hate)      : 0 = not hate,      1 = hate speech (implicit)
  Column 3 (offensive) : 0 = not offensive,  1 = offensive speech

  Combined tri-class mapping used in this project:
    hate=1  (regardless of offensive)  -->  "hate"       (label 2)
    hate=0, offensive=1                -->  "offensive"  (label 1)
    hate=0, offensive=0                -->  "normal"     (label 0)

This script:
  - Merges the two JSON files on the shared `index` key
  - Cleans and normalises tweet text
  - Derives the 3-class label
  - Performs stratified 70 / 15 / 15 train / val / test split
  - Saves processed CSVs + class_weights.json to data/processed/
"""

import os
import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15

RAW_DIR     = "data/raw"
OUTPUT_DIR  = "data/processed"

TEXT_FILE   = os.path.join(RAW_DIR, "THOS_Dataset_Text.json")
LABEL_FILE  = os.path.join(RAW_DIR, "THOS_Dataset.json")

# Final label mapping
LABEL2ID = {"normal": 0, "offensive": 1, "hate": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load JSON files ────────────────────────────────────────────────────────────
def load_json(path: str) -> list:
    """
    Load a JSON file that is either:
      - a top-level list:  [{...}, {...}, ...]
      - a dict wrapping a list:  {"data": [{...}, ...]}
    """
    print(f"[INFO] Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"Unexpected JSON structure in {path}. "
                     f"Top-level type: {type(raw)}")


#merging labels
def merge_datasets(text_records: list, label_records: list):
    """
    Join THOS_Dataset_Text.json and THOS_Dataset.json on their shared index field.

    THOS_Dataset_Text.json rows:
        {"index": 42, "text": "some tweet ..."}

    THOS_Dataset.json rows:
        {"index": 42, <hate_col>, <offensive_col>, <topic cols> ...}

    Returns: (merged_df, hate_col_name, offensive_col_name)
    """
    df_text  = pd.DataFrame(text_records)
    df_label = pd.DataFrame(label_records)

    print(f"[INFO] Text  records : {len(df_text):,}")
    print(f"[INFO] Label records : {len(df_label):,}")
    print(f"[INFO] Label columns : {df_label.columns.tolist()}")

    #detecting shared columns
    index_candidates = ["index", "id", "ID", "tweet_id"]

    idx_text  = next((c for c in index_candidates if c in df_text.columns),
                     df_text.columns[0])
    idx_label = next((c for c in index_candidates if c in df_label.columns),
                     df_label.columns[0])

    df_text  = df_text.rename(columns={idx_text:  "index"})
    df_label = df_label.rename(columns={idx_label: "index"})

    df = pd.merge(df_text, df_label, on="index", how="inner")
    print(f"[INFO] After merge   : {len(df):,} rows")

    #text columns
    text_candidates = ["text", "tweet", "content", "Tweet", "Text"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        raise ValueError(f"Cannot find text column. Columns available: {df.columns.tolist()}")
    df = df.rename(columns={text_col: "text"})

    # detecting hate

    hate_candidates      = ["hate",      "Hate",      "is_hate",      "hate_speech", "col2", "column2", "Implicit"]
    offensive_candidates = ["offensive", "Offensive", "is_offensive", "col3",        "column3", "Explicit"]

    hate_col = next((c for c in hate_candidates if c in df.columns), None)
    off_col  = next((c for c in offensive_candidates if c in df.columns), None)


    if hate_col is None or off_col is None:
        non_meta = [c for c in df_label.columns if c not in ("index", "text")]
        print(f"[WARN] Auto-detect failed. Trying positional fallback.")
        print(f"       Non-meta label columns (first 5): {non_meta[:5]}")
        if len(non_meta) >= 2:
            hate_col = non_meta[0]
            off_col  = non_meta[1]
        else:
            raise ValueError(
                "Cannot detect hate/offensive columns automatically.\n"
                f"All columns: {df.columns.tolist()}\n"
                "Set hate_col and off_col manually in derive_label()."
            )

    print(f"[INFO] Hate column       : '{hate_col}'")
    print(f"[INFO] Offensive column  : '{off_col}'")

    return df, hate_col, off_col



def derive_label(row, hate_col: str, off_col: str) -> int:
    """
    Priority order: hate > offensive > normal
    Handles int (0/1), float (0.0/1.0), and string ('0'/'1') cell values.
    Returns -1 for unparseable rows (will be dropped).
    """
    try:
        hate = int(float(str(row[hate_col]).strip()))
        off  = int(float(str(row[off_col]).strip()))
    except (ValueError, KeyError):
        return -1

    if hate != 0:
        return LABEL2ID["hate"]
    if off != 0:
        return LABEL2ID["offensive"]
    return LABEL2ID["normal"]



def clean_tweet(text: str) -> str:
    """
    Normalise raw tweet text for BERT tokenisation.

    THOS already replaces usernames with <user> and links with <url>.
    We further normalise these and handle any raw tokens that slipped through.

    Steps:
      1. Lowercase
      2. <user> / @mention  -->  [USER]
      3. <url> / raw URL    -->  [URL]
      4. Hashtag #word      -->  word  (remove # symbol)
      5. HTML entities
      6. Collapse whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # THOS-native placeholders
    text = text.replace("<user>", "[USER]")
    text = text.replace("<url>",  "[URL]")

    # Raw mentions / URLs that may remain in some rows
    text = re.sub(r"http\S+|www\.\S+", "[URL]",  text)
    text = re.sub(r"@\w+",             "[USER]", text)

    # Hashtags: strip the # sign but keep the word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Common HTML entities
    for entity, char in [("&amp;","&"),("&lt;","<"),("&gt;",">"),
                          ("&quot;",'"'),("&apos;","'")]:
        text = text.replace(entity, char)

    # Collapse any repeated whitespace / newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text



def class_report(df: pd.DataFrame, split_name: str = "full") -> None:
    counts = Counter(df["label"])
    total  = len(df)
    print(f"\n{'─'*50}")
    print(f"  Class distribution  [{split_name}]   n={total:,}")
    print(f"{'─'*50}")
    for lid in sorted(ID2LABEL):
        name = ID2LABEL[lid]
        n    = counts.get(lid, 0)
        pct  = 100 * n / total if total else 0
        bar  = "█" * int(pct / 2)
        print(f"  {name:>10} ({lid})  {n:5,}  {pct:5.1f}%  {bar}")
    print(f"{'─'*50}")



def split_and_save(df: pd.DataFrame) -> None:
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        split[["text", "label"]].to_csv(path, index=False)
        print(f"[INFO] Saved {name} --> {path}  ({len(split):,} rows)")
        class_report(split, name)


def compute_class_weights(df: pd.DataFrame) -> dict:
    """
    Inverse-frequency weights, normalised so they sum to num_classes.
    Use with torch.nn.CrossEntropyLoss(weight=...) or FocalLoss(alpha=...).
    """
    counts  = np.array([Counter(df["label"]).get(i, 1) for i in range(len(LABEL2ID))])
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(LABEL2ID)
    w_dict  = {ID2LABEL[i]: round(float(w), 6) for i, w in enumerate(weights)}
    print(f"[INFO] Class weights (train): {w_dict}")
    return w_dict



def main():
    print("=" * 55)
    print("  STEP 1 — Data Preprocessing (THOS local JSON)")
    print("=" * 55)

    # Validate files exist
    for path in [TEXT_FILE, LABEL_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n[ERROR] File not found: {path}\n\n"
                "Download both files from https://github.com/mohaimeed/THOS\n"
                f"and place them in:  {RAW_DIR}/\n"
                "  THOS_Dataset_Text.json\n"
                "  THOS_Dataset.json\n"
            )

    # 1. Load raw JSON
    text_records  = load_json(TEXT_FILE)
    label_records = load_json(LABEL_FILE)

    # 2. Merge on shared index
    df, hate_col, off_col = merge_datasets(text_records, label_records)

    # 3. Derive 3-class label
    print("[INFO] Deriving 3-class labels ...")
    df["label"] = df.apply(lambda r: derive_label(r, hate_col, off_col), axis=1)
    before = len(df)
    df = df[df["label"] >= 0].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[WARN] Dropped {dropped} rows with unrecognisable labels")

    # 4. Clean text
    print("[INFO] Cleaning tweet text ...")
    df["text"] = df["text"].apply(clean_tweet)
    df = df[df["text"].str.len() > 0].copy()
    print(f"[INFO] Total usable rows after cleaning: {len(df):,}")

    # 5. Full-dataset class report
    class_report(df, "full")

    # 6. Stratified split & save CSVs
    split_and_save(df)

    # 7. Class weights (computed from training split only — no data leakage)
    train_df = pd.read_csv(os.path.join(OUTPUT_DIR, "train.csv"))
    weights  = compute_class_weights(train_df)
    with open(os.path.join(OUTPUT_DIR, "class_weights.json"), "w") as f:
        json.dump(weights, f, indent=2)
    print(f"[INFO] Saved class_weights.json --> {OUTPUT_DIR}/class_weights.json")

    # 8. Quick sanity check — 2 random examples per class from training set
    print("\n[SANITY CHECK] 2 random examples per class from train split:")
    for lid, name in ID2LABEL.items():
        subset = train_df[train_df["label"] == lid]
        sample = subset.sample(min(2, len(subset)), random_state=RANDOM_SEED)
        print(f"\n  ── {name.upper()} (label={lid}) ──")
        for _, row in sample.iterrows():
            print(f"    {str(row['text'])[:110]}")

    print(f"\n[DONE] Step 1 complete. All files saved to: {OUTPUT_DIR}/")
    print("       Next: run step2_baseline_bert.py")


if __name__ == "__main__":
    main()
