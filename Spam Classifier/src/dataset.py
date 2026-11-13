"""
Dataset Acquisition & Ingestion Pipeline
Author: Senior ML Engineer
Project: Spam Email Classifier - End-to-End Machine Learning System

This module is responsible for:
1. Downloading the real public dataset (UCI Machine Learning Repository).
2. Detecting and normalizing custom datasets placed in data/raw.
3. Conducting Exploratory Data Analysis (EDA) and reporting statistics.
4. Cleaning missing values and duplicates.
5. Saving raw and clean processed datasets.
"""

import os
import io
import zipfile
import urllib.request
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "spam_dataset.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")

# UCI Machine Learning Repository: SMS Spam Collection Dataset
UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"


def download_uci_dataset(dest_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Downloads the SMS Spam Collection dataset directly from the official
    UCI Machine Learning Repository and saves it as a CSV.
    
    Returns:
        pd.DataFrame: Raw dataset with columns ['label', 'text'].
    """
    logger.info("Attempting to download dataset from UCI Machine Learning Repository...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    request = urllib.request.Request(UCI_DATASET_URL, headers=headers)
    
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()
        logger.info(f"Downloaded {len(archive_bytes)} bytes from UCI.")
        
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        file_list = zf.namelist()
        logger.info(f"Archive contents: {file_list}")
        if "SMSSpamCollection" not in file_list:
            raise FileNotFoundError("Expected 'SMSSpamCollection' file not found in zip archive.")
        
        raw_text = zf.read("SMSSpamCollection").decode("utf-8", errors="ignore")

    # The file is tab-delimited: <label>\t<text>
    records = []
    for line in raw_text.strip().split("\n"):
        parts = line.split("\t", 1)
        if len(parts) == 2:
            records.append((parts[0].strip(), parts[1].strip()))

    df = pd.DataFrame(records, columns=["label", "text"])
    df.to_csv(dest_path, index=False)
    logger.info(f"Saved raw dataset to: {dest_path} (Total records: {len(df)})")
    return df


def detect_and_normalize_local_dataset() -> pd.DataFrame:
    """
    Scans data/raw/ for any user-provided CSV dataset and normalizes
    column names to standard ['label', 'text'].
    
    Supports column schemes:
    - ['label', 'text']
    - ['v1', 'v2'] (Kaggle spam format)
    - ['Category', 'Message']
    - ['Mail', 'Text']
    - ['label_num', 'text']
    """
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        return None

    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        return None

    # Prefer spam_dataset.csv if it already exists, else pick first csv
    target_file = "spam_dataset.csv" if "spam_dataset.csv" in csv_files else csv_files[0]
    full_path = os.path.join(RAW_DATA_DIR, target_file)
    logger.info(f"Loading local dataset from: {full_path}")
    
    # Try different encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(full_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode file {full_path} with standard encodings.")

    # Column mapping logic
    cols = {c.strip().lower(): c for c in df.columns}
    label_col, text_col = None, None

    if "label" in cols:
        label_col = cols["label"]
    elif "v1" in cols:
        label_col = cols["v1"]
    elif "category" in cols:
        label_col = cols["category"]
    elif "mail" in cols:
        label_col = cols["mail"]
    elif "target" in cols:
        label_col = cols["target"]

    if "text" in cols:
        text_col = cols["text"]
    elif "v2" in cols:
        text_col = cols["v2"]
    elif "message" in cols:
        text_col = cols["message"]
    elif "email" in cols:
        text_col = cols["email"]

    if label_col is None or text_col is None:
        raise ValueError(
            f"Could not automatically detect label and text columns in {full_path}. "
            f"Columns found: {list(df.columns)}. Please ensure columns are named 'label' and 'text'."
        )

    normalized_df = pd.DataFrame({
        "label": df[label_col].astype(str).str.strip().str.lower(),
        "text": df[text_col].astype(str)
    })

    # Standardize label values (0/1 or ham/spam)
    label_map = {
        "0": "ham", "1": "spam",
        "0.0": "ham", "1.0": "spam",
        "ham": "ham", "spam": "spam"
    }
    normalized_df["label"] = normalized_df["label"].map(lambda x: label_map.get(str(x).lower(), str(x).lower()))
    
    # Filter to only valid ham and spam classes
    normalized_df = normalized_df[normalized_df["label"].isin(["ham", "spam"])]
    return normalized_df


def load_or_fetch_dataset() -> pd.DataFrame:
    """
    Loads existing raw dataset or automatically downloads it.
    """
    # 1. Try local dataset first
    if os.path.exists(RAW_DATA_PATH):
        try:
            df = detect_and_normalize_local_dataset()
            if df is not None and len(df) > 0:
                logger.info(f"Loaded existing raw dataset from disk: {RAW_DATA_PATH}")
                return df
        except Exception as e:
            logger.warning(f"Error loading existing raw dataset: {e}. Will re-download.")

    # 2. Download from UCI repository
    try:
        df = download_uci_dataset(RAW_DATA_PATH)
        return df
    except Exception as e:
        logger.error(f"Failed to automatically download dataset: {e}")
        logger.info(
            "\n" + "=" * 60 + "\n"
            "MANUAL DATASET INSTRUCTIONS:\n"
            "1. Download a spam dataset (e.g. from Kaggle or UCI).\n"
            f"2. Place the CSV file into: {RAW_DATA_DIR}\n"
            "3. Name it 'spam_dataset.csv' with columns: 'label', 'text'\n"
            f"   (Where label is 'ham' or 'spam').\n"
            "=" * 60
        )
        raise


def explore_dataset(df: pd.DataFrame) -> dict:
    """
    Performs Exploratory Data Analysis (EDA) on the dataset and prints summary.
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 60)
    
    total_samples = len(df)
    null_counts = df.isnull().sum().to_dict()
    duplicates_count = int(df.duplicated(subset=["text"]).sum())
    
    class_counts = df["label"].value_counts().to_dict()
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)
    
    ham_pct = (ham_count / total_samples * 100) if total_samples > 0 else 0
    spam_pct = (spam_count / total_samples * 100) if total_samples > 0 else 0

    # Text length stats
    df_temp = df.copy()
    df_temp["char_len"] = df_temp["text"].astype(str).str.len()
    avg_len_ham = df_temp[df_temp["label"] == "ham"]["char_len"].mean() if ham_count > 0 else 0
    avg_len_spam = df_temp[df_temp["label"] == "spam"]["char_len"].mean() if spam_count > 0 else 0

    stats = {
        "total_records": total_samples,
        "null_values": null_counts,
        "duplicate_texts": duplicates_count,
        "class_distribution": {
            "ham": {"count": ham_count, "percentage": round(ham_pct, 2)},
            "spam": {"count": spam_count, "percentage": round(spam_pct, 2)}
        },
        "avg_char_length": {
            "ham": round(float(avg_len_ham), 2),
            "spam": round(float(avg_len_spam), 2)
        }
    }

    logger.info(f"Total Records: {total_samples}")
    logger.info(f"Null Values: {null_counts}")
    logger.info(f"Duplicate Messages: {duplicates_count}")
    logger.info(f"Class Breakdown: Ham={ham_count} ({ham_pct:.2f}%), Spam={spam_count} ({spam_pct:.2f}%)")
    logger.info(f"Average Message Length: Ham={avg_len_ham:.1f} chars, Spam={avg_len_spam:.1f} chars")
    logger.info("=" * 60)

    return stats


def clean_and_process_dataset(df: pd.DataFrame, dest_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """
    Cleans raw data by removing nulls, empty texts, and duplicates,
    then persists the cleaned dataset.
    """
    logger.info("Starting dataset cleaning...")
    initial_count = len(df)
    
    # 1. Drop missing records
    clean_df = df.dropna(subset=["label", "text"]).copy()
    
    # 2. Trim whitespace and filter out empty strings
    clean_df["text"] = clean_df["text"].astype(str).str.strip()
    clean_df["label"] = clean_df["label"].astype(str).str.strip().str.lower()
    clean_df = clean_df[clean_df["text"].str.len() > 0]
    clean_df = clean_df[clean_df["label"].isin(["ham", "spam"])]
    
    # 3. Remove duplicate records
    clean_df = clean_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    final_count = len(clean_df)
    removed_count = initial_count - final_count
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    clean_df.to_csv(dest_path, index=False)
    
    logger.info(
        f"Cleaning Complete: {final_count} records retained ({removed_count} removed). "
        f"Saved to: {dest_path}"
    )
    return clean_df


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    End-to-end dataset acquisition and processing pipeline.
    """
    raw_df = load_or_fetch_dataset()
    eda_stats = explore_dataset(raw_df)
    clean_df = clean_and_process_dataset(raw_df)
    return raw_df, clean_df, eda_stats


if __name__ == "__main__":
    run_pipeline()
