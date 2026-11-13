"""
Inference & Prediction Pipeline
Author: Senior ML Engineer
Project: Spam Email Classifier - End-to-End Machine Learning System

This module provides high-throughput, low-latency email classification
by loading the model and TF-IDF vectorizer once into memory.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


class SpamClassifierService:
    """
    Singleton-style predictor service that holds the model and vectorizer
    in memory for real-time inference without reloading overhead.
    """
    _instance = None
    _model = None
    _vectorizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpamClassifierService, cls).__new__(cls)
            cls._instance._load_artifacts()
        return cls._instance

    def _load_artifacts(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Model or Vectorizer artifacts missing at {MODELS_DIR}. "
                "Please run 'python src/train.py' before starting the service."
            )
        logger.info(f"Loading champion model from {MODEL_PATH}...")
        self._model = joblib.load(MODEL_PATH)
        logger.info(f"Loading TF-IDF vectorizer from {VECTORIZER_PATH}...")
        self._vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("SpamClassifierService initialized successfully.")

    def predict(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Classifies an input text as Spam or Ham (Legitimate).

        Args:
            text: Raw email or message text.

        Returns:
            Dictionary containing prediction outcome, confidence, and metadata.
        """
        if text is None or not isinstance(text, str) or not text.strip():
            return {
                "success": False,
                "error": "Email text cannot be empty or blank.",
                "is_spam": False,
                "label": "Invalid",
                "confidence": 0.0
            }

        raw_trimmed = text.strip()
        cleaned = clean_text(raw_trimmed)

        # In case text had only symbols with no alphanumeric words (e.g., "$$$ !!!")
        if not cleaned:
            # Fallback: if text consists solely of spam-like symbols (e.g. $$$), classify conservatively
            features_vec = self._vectorizer.transform([raw_trimmed.lower()])
        else:
            features_vec = self._vectorizer.transform([cleaned])

        # Inference
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(features_vec)[0]
            ham_prob = float(proba[0])
            spam_prob = float(proba[1])
        else:
            # Decision function fallback
            dec = float(self._model.decision_function(features_vec)[0])
            spam_prob = 1.0 / (1.0 + (2.71828 ** (-dec)))
            ham_prob = 1.0 - spam_prob

        prediction_label = 1 if spam_prob >= 0.5 else 0
        is_spam = bool(prediction_label == 1)
        label_str = "Spam" if is_spam else "Ham"
        
        # Confidence score is the winning class's probability
        confidence_val = spam_prob if is_spam else ham_prob
        confidence_pct = round(confidence_val * 100.0, 2)

        return {
            "success": True,
            "is_spam": is_spam,
            "label": label_str,
            "confidence": confidence_pct,
            "confidence_score": round(confidence_val, 4),
            "probabilities": {
                "ham": round(ham_prob, 4),
                "spam": round(spam_prob, 4)
            },
            "cleaned_tokens": cleaned,
            "char_count": len(raw_trimmed),
            "word_count": len(raw_trimmed.split()),
            "preview": raw_trimmed[:120] + ("..." if len(raw_trimmed) > 120 else "")
        }


# Module-level convenience function
_predictor_instance = None

def predict_email(text: str) -> Dict[str, Any]:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = SpamClassifierService()
    return _predictor_instance.predict(text)


if __name__ == "__main__":
    test_cases = [
        "Congratulations! You won a $1,000 Walmart Gift Card. Click https://claim-prize.net now to claim!",
        "Subject: Q3 Financial Planning Meeting. Let's review the spreadsheet tomorrow at 10 AM in Conference Room B.",
        "URGENT: Your account access has been suspended due to security risks. Log in now at http://secure-verify.com",
        "Hey, can you send me the recipe for the lasagna you made last weekend? It was delicious!"
    ]

    print("=" * 70)
    print("INFERENCE PIPELINE VERIFICATION")
    print("=" * 70)
    for sample in test_cases:
        res = predict_email(sample)
        print(f"\n[TEXT]:       {sample[:75]}...")
        print(f"[PREDICTION]: {res['label']} ({res['confidence']}% confidence)")
        print(f"[PROB]:       Ham: {res['probabilities']['ham']} | Spam: {res['probabilities']['spam']}")
