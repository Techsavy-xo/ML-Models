"""
Machine Learning Training & Model Selection Pipeline
Author: Senior ML Engineer
Project: Spam Email Classifier - End-to-End Machine Learning System

This module:
1. Loads the preprocessed dataset.
2. Applies text preprocessing to train/test splits.
3. Fits TF-IDF vectorizer on training data ONLY (preventing data leakage).
4. Trains 4 distinct ML architectures:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (with calibrated probabilities)
5. Compares all 4 models across Accuracy, Precision, Recall, F1, and Confusion Matrix.
6. Selects the champion model based on F1-score & Precision on Spam.
7. Saves the champion model and vectorizer using Joblib, plus a JSON metrics record.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from src.preprocessing import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")


def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the cleaned dataset and checks integrity.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Please execute 'python src/dataset.py' first."
        )
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset from {data_path} (Shape: {df.shape})")
    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocesses text and splits into train and test sets with stratification.
    Prevents data leakage by keeping train and test sets strictly separate.
    """
    logger.info("Cleaning raw text samples...")
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Map classes: 0 for ham (legitimate), 1 for spam
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    
    # Filter out empty cleaned strings if any
    valid_mask = df["clean_text"].str.strip().str.len() > 0
    df = df[valid_mask].reset_index(drop=True)
    
    logger.info(f"Valid preprocessed samples for modeling: {len(df)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["target"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"]
    )
    
    logger.info(
        f"Train split: {len(X_train)} samples ({y_train.sum()} spam) | "
        f"Test split: {len(X_test)} samples ({y_test.sum()} spam)"
    )
    return X_train, X_test, y_train, y_test


def build_vectorizer(max_features: int = 4000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Instantiates TF-IDF Vectorizer with unigrams and bigrams.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True
    )


def get_candidate_models() -> dict:
    """
    Defines the candidate model dictionary.
    All models output well-calibrated confidence probabilities.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.2),
        "Logistic Regression": LogisticRegression(
            C=2.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=SVC(kernel="linear", C=1.0, random_state=42, class_weight="balanced"),
            cv=3
        )
    }


def train_and_evaluate_all():
    """
    Executes the training pipeline, compares all 4 algorithms,
    saves the champion model, vectorizer, and comparison metrics JSON.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load & split
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 2. Fit TF-IDF only on training data
    logger.info("Fitting TF-IDF Vectorizer on training data...")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)} features")
    
    # 3. Train and benchmark candidate models
    candidates = get_candidate_models()
    results = {}
    trained_estimators = {}
    
    best_name = None
    best_f1 = -1.0
    best_estimator = None
    
    logger.info("=" * 70)
    logger.info(f"{'Model':<26} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8}")
    logger.info("=" * 70)

    for name, model in candidates.items():
        # Train
        model.fit(X_train_vec, y_train)
        trained_estimators[name] = model
        
        # Predict on holdout test set
        y_pred = model.predict(X_test_vec)
        
        # Predict probabilities / decision scores for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test_vec)[:, 1]
        else:
            y_scores = model.decision_function(X_test_vec)

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_scores))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Compute smooth downsampled ROC Curve coordinates (FPR vs TPR) for Chart.js
        fpr_raw, tpr_raw, _ = roc_curve(y_test, y_scores)
        # Uniform sampling across 15 target FPR values
        target_fprs = np.linspace(0.0, 1.0, 21)
        roc_points = []
        for tf in target_fprs:
            # Interpolate TPR at target FPR
            interp_tpr = float(np.interp(tf, fpr_raw, tpr_raw))
            roc_points.append({"x": round(float(tf), 4), "y": round(interp_tpr, 4)})

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
            "roc_curve": roc_points
        }
        
        logger.info(f"{name:<26} | {acc:<8.4f} | {prec:<9.4f} | {rec:<7.4f} | {f1:<8.4f} | AUC: {roc_auc:.4f}")
        
        # Selection criterion: highest F1-score with high precision (minimizing false positives)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_estimator = model

    logger.info("=" * 70)
    logger.info(f"Champion Model Selected: '{best_name}' (F1-Score: {best_f1:.4f})")
    
    # 4. Save champion model and vectorizer
    joblib.dump(best_estimator, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    logger.info(f"Saved champion model to: {BEST_MODEL_PATH}")
    logger.info(f"Saved TF-IDF vectorizer to: {VECTORIZER_PATH}")
    
    # 5. Persist comparison metrics for dashboard
    comparison_payload = {
        "timestamp": datetime.now().isoformat(),
        "total_training_samples": len(X_train),
        "total_test_samples": len(X_test),
        "champion_model": best_name,
        "models": results
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    logger.info(f"Saved model comparison metrics to: {METRICS_JSON_PATH}")
    
    return best_name, results


if __name__ == "__main__":
    train_and_evaluate_all()
