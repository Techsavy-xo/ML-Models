"""
High-Throughput Bulk Email Classification Pipeline
Author: Senior ML Engineer & Backend Architect
Project: Spam Email Classifier - Advanced Enterprise Upgrade

Enables batch processing of CSV files containing hundreds or thousands of emails.
Utilizes matrix operations for high throughput and generates downloadable CSV reports.
"""

import io
import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import SpamClassifierService
from src.preprocessing import clean_text
from src.phishing_analyzer import analyze_phishing_threats
from src.risk_scoring import compute_risk_score

logger = logging.getLogger(__name__)


def detect_email_column(df: pd.DataFrame) -> str:
    """
    Auto-detects the column containing email text in an uploaded CSV.
    Compatible with Pandas 1.x, 2.x, and 3.x string dtypes.
    """
    cols = {str(c).strip().lower(): c for c in df.columns}
    
    candidates = [
        "email", "text", "message", "body", "email_text", "mail", "content",
        "subject", "mail_body", "email_body", "raw_text", "sms"
    ]
    for cand in candidates:
        if cand in cols:
            return cols[cand]

    # Substring matching
    for col_lower, orig_col in cols.items():
        if any(keyword in col_lower for keyword in ["mail", "text", "msg", "body"]):
            return orig_col

    # Fallback to the first string or object column with highest average character length
    str_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == "object"]
    if str_cols:
        best_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        return best_col

    raise ValueError(
        f"Could not identify an email text column. Columns found: {list(df.columns)}. "
        "Please include a column named 'email', 'text', 'message', or 'body'."
    )


def process_bulk_dataframe(
    df: pd.DataFrame,
    email_col: Optional[str] = None,
    max_rows: int = 2500
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs high-speed batch classification and threat analysis on a DataFrame.

    Args:
        df: Input pandas DataFrame.
        email_col: Name of column containing email text (auto-detected if None).
        max_rows: Maximum rows allowed per bulk batch to prevent memory exhaustion.

    Returns:
        Tuple of (Summary dictionary, Enriched DataFrame with predictions).
    """
    if df.empty:
        raise ValueError("Uploaded CSV file is empty.")

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    if email_col is None or email_col not in df.columns:
        email_col = detect_email_column(df)

    logger.info(f"Processing bulk batch: {len(df)} emails using column '{email_col}'...")

    # Initialize predictor
    predictor = SpamClassifierService()
    model = predictor._model
    vectorizer = predictor._vectorizer

    raw_texts = df[email_col].fillna("").astype(str).tolist()

    # 1. Batch NLP Preprocessing
    cleaned_texts = [clean_text(t) for t in raw_texts]

    # 2. Batch Matrix TF-IDF Transformation
    X_vec = vectorizer.transform(cleaned_texts)

    # 3. Batch Probabilistic Model Inference
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_vec)
        ham_probs = probabilities[:, 0]
        spam_probs = probabilities[:, 1]
    else:
        dec = model.decision_function(X_vec)
        spam_probs = 1.0 / (1.0 + np.exp(-dec))
        ham_probs = 1.0 - spam_probs

    # 4. Generate Row-by-Row Forensics & Risk Scores
    results = []
    total_spam = 0
    total_ham = 0
    high_risk_count = 0

    for i, raw_text in enumerate(raw_texts):
        s_prob = float(spam_probs[i])
        h_prob = float(ham_probs[i])
        is_spam = bool(s_prob >= 0.5)
        label = "Spam" if is_spam else "Ham"
        conf = round((s_prob if is_spam else h_prob) * 100.0, 2)

        if is_spam:
            total_spam += 1
        else:
            total_ham += 1

        # Heuristic Phishing Check
        threat_info = analyze_phishing_threats(raw_text)
        h_score = threat_info["threat_score"]
        is_phish = threat_info["is_phishing_suspect"]

        # Composite Risk Score
        risk_meta = compute_risk_score(s_prob, h_score, is_phish)
        risk_val = risk_meta["risk_score"]
        if risk_val >= 75:
            high_risk_count += 1

        preview = raw_text.strip().replace("\n", " ").replace("\r", " ")
        if len(preview) > 90:
            preview = preview[:90] + "..."

        results.append({
            "id": i + 1,
            "preview": preview,
            "classification": label,
            "is_spam": is_spam,
            "confidence_percent": conf,
            "spam_probability": round(s_prob, 4),
            "ham_probability": round(h_prob, 4),
            "risk_score": risk_val,
            "risk_level": risk_meta["risk_level"],
            "phishing_indicators_count": threat_info["total_indicators_count"],
            "top_threat_flag": threat_info["url_analysis"]["flags"][0] if threat_info["url_analysis"]["flags"]
                               else (threat_info["urgency_indicators"][0]["pattern"] if threat_info["urgency_indicators"] else "None")
        })

    enriched_df = pd.DataFrame(results)

    total = len(results)
    summary = {
        "total_emails": total,
        "spam_count": total_spam,
        "ham_count": total_ham,
        "spam_percentage": round((total_spam / total * 100.0), 2) if total > 0 else 0.0,
        "high_risk_count": high_risk_count,
        "avg_confidence": round(float(enriched_df["confidence_percent"].mean()), 2) if total > 0 else 0.0,
        "records": results[:100]  # Return first 100 for immediate UI display
    }

    return summary, enriched_df


def generate_export_csv(enriched_df: pd.DataFrame) -> str:
    """
    Exports the classification results to a standard CSV string.
    """
    export_cols = [
        "id", "preview", "classification", "confidence_percent",
        "risk_score", "risk_level", "phishing_indicators_count", "top_threat_flag"
    ]
    cols_to_use = [c for c in export_cols if c in enriched_df.columns]
    return enriched_df[cols_to_use].to_csv(index=False)


if __name__ == "__main__":
    sample_data = {
        "email_text": [
            "URGENT: Your PayPal account has been suspended! Click http://192.168.1.1/login within 24h.",
            "Hey David, are we still meeting tomorrow for sprint planning at 10 AM?",
            "WINNER!! You have won a $1,000 Walmart Gift Card! Call now to claim free prize.",
            "Please find attached the financial review spreadsheet from yesterday's sync."
        ]
    }
    test_df = pd.DataFrame(sample_data)
    summary, enriched = process_bulk_dataframe(test_df)
    print("=" * 65)
    print("BULK PROCESSOR MODULE VERIFICATION")
    print("=" * 65)
    print(f"Total Processed: {summary['total_emails']}")
    print(f"Spam: {summary['spam_count']} | Ham: {summary['ham_count']} ({summary['spam_percentage']}%)")
    print(f"High Risk Items: {summary['high_risk_count']}")
    print("\nPreview of Enriched Output:\n", enriched[["id", "classification", "confidence_percent", "risk_score", "risk_level"]])
