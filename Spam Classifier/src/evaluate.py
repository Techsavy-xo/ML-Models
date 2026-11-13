"""
Model Evaluation & Diagnostics Module
Author: Senior ML Engineer
Project: Spam Email Classifier - End-to-End Machine Learning System

This module loads the trained models and test data to generate:
- Comprehensive classification reports (Precision, Recall, F1 per class)
- Confusion matrix breakdown (True Negatives, False Positives, False Negatives, True Positives)
- Performance summary across classes
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from src.preprocessing import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


def evaluate_saved_model(
    model_path: str = MODEL_PATH,
    vectorizer_path: str = VECTORIZER_PATH,
    data_path: str = DATA_PATH
) -> dict:
    """
    Evaluates the persisted champion model on holdout test data.
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "Trained model or vectorizer not found. Run 'python src/train.py' first."
        )

    logger.info(f"Loading champion model from: {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading vectorizer from: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)

    logger.info(f"Loading test data from: {data_path}")
    df = pd.read_csv(data_path)
    df["clean_text"] = df["text"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    
    valid_mask = df["clean_text"].str.strip().str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    # Use exact same split parameters to reproduce test set
    _, X_test, _, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=0.2,
        random_state=42,
        stratify=df["target"]
    )

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_vec)[:, 1]
    else:
        y_scores = model.decision_function(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_scores)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=["Ham (Legitimate)", "Spam"],
        output_dict=True
    )
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Ham (Legitimate)", "Spam"]
    )

    print("\n" + "=" * 65)
    print("CHAMPION MODEL EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"ROC-AUC Score:     {roc_auc:.4f} ({roc_auc * 100:.2f}%)")
    print(f"Spam Precision:    {prec * 100:.2f}% (Minimizes false alarms on legitimate emails)")
    print(f"Spam Recall:       {rec * 100:.2f}% (Percentage of true spams caught)")
    print(f"Spam F1-Score:     {f1 * 100:.2f}%\n")
    print("Confusion Matrix Breakdown:")
    print(f"  • True Negatives  (Legitimate correctly marked): {tn}")
    print(f"  • False Positives (Legitimate falsely flagged):  {fp}")
    print(f"  • False Negatives (Spam missed):                 {fn}")
    print(f"  • True Positives  (Spam correctly caught):       {tp}\n")
    print("Full Classification Report:")
    print(report_text)
    print("=" * 65 + "\n")

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "classification_report": report_dict
    }


if __name__ == "__main__":
    evaluate_saved_model()
