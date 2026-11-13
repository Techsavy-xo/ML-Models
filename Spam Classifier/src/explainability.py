"""
Explainable AI (XAI) & Feature Attribution Engine
Author: Senior ML Engineer
Project: Spam Email Classifier - Advanced Enterprise Upgrade

This module provides model explainability:
1. Exact feature attribution (Linear Shapley values: C_i = w_i * x_i).
2. Word-level contribution breakdown (Positive / Spam vs. Negative / Ham).
3. Visual HTML markup generator with span highlighting for suspicious and reassuring words.
4. Natural language explanation summarizing why the model made its decision.
"""

import os
import sys
import re
import html
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import joblib
from nltk.stem import PorterStemmer

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "spam_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")


class EmailExplainerService:
    """
    Singleton service that computes exact feature attributions and visual
    in-text explanations for predictions made by the Linear SVM classifier.
    """
    _instance = None
    _model = None
    _vectorizer = None
    _weights = None
    _intercept = 0.0
    _feature_names = None
    _feature_index_map = None
    _stemmer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmailExplainerService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"Model or Vectorizer artifacts missing at {MODELS_DIR}."
            )
        
        logger.info(f"Initializing EmailExplainerService from {MODEL_PATH}...")
        self._model = joblib.load(MODEL_PATH)
        self._vectorizer = joblib.load(VECTORIZER_PATH)
        self._stemmer = PorterStemmer()
        
        # Extract feature names & build fast reverse lookup map
        self._feature_names = self._vectorizer.get_feature_names_out()
        self._feature_index_map = {feat: idx for idx, feat in enumerate(self._feature_names)}

        # Extract weights from CalibratedClassifierCV or base linear estimator
        if hasattr(self._model, "calibrated_classifiers_"):
            # Average the coefficient vector across calibration folds
            fold_weights = [
                clf.estimator.coef_.toarray().ravel()
                for clf in self._model.calibrated_classifiers_
            ]
            fold_intercepts = [
                float(clf.estimator.intercept_[0])
                for clf in self._model.calibrated_classifiers_
            ]
            self._weights = np.mean(fold_weights, axis=0)
            self._intercept = float(np.mean(fold_intercepts))
        elif hasattr(self._model, "coef_"):
            self._weights = self._model.coef_.ravel()
            self._intercept = float(self._model.intercept_[0])
        else:
            raise AttributeError("Loaded model does not expose linear coefficients.")

        logger.info(
            f"EmailExplainerService ready. Intercept: {self._intercept:.4f}, "
            f"Vocabulary: {len(self._feature_names)} features."
        )

    def explain(self, text: str, max_features: int = 8) -> Dict[str, Any]:
        """
        Computes detailed feature attributions, word-level highlights, and explanations.

        Args:
            text: Raw input email text.
            max_features: Maximum number of top positive and negative features to return.

        Returns:
            Dictionary containing:
            - top_spam_features: list of words pushing toward Spam
            - top_ham_features: list of words pushing toward Legitimate
            - net_score: decision value z
            - annotated_html: original text with highlighted tokens
            - natural_explanation: conversational summary of key factors
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "success": False,
                "error": "Input text cannot be empty.",
                "top_spam_features": [],
                "top_ham_features": [],
                "annotated_html": "",
                "natural_explanation": "No text provided for analysis."
            }

        cleaned = clean_text(text)
        vec_sparse = self._vectorizer.transform([cleaned])
        indices = vec_sparse.indices
        tfidf_values = vec_sparse.data

        feature_contributions: List[Dict[str, Any]] = []
        token_score_lookup: Dict[str, float] = {}

        for idx, val in zip(indices, tfidf_values):
            feat_name = self._feature_names[idx]
            w = float(self._weights[idx])
            contrib = float(w * val)
            
            # Map single tokens for in-text highlight matching
            for sub_tok in feat_name.split():
                token_score_lookup[sub_tok] = token_score_lookup.get(sub_tok, 0.0) + contrib

            feature_contributions.append({
                "feature": feat_name,
                "tfidf": round(float(val), 4),
                "weight": round(w, 4),
                "contribution": round(contrib, 4),
                "direction": "spam" if contrib > 0 else "ham"
            })

        # Separate spam contributors (positive) and ham anchors (negative)
        spam_features = sorted(
            [f for f in feature_contributions if f["contribution"] > 0],
            key=lambda x: x["contribution"],
            reverse=True
        )
        ham_features = sorted(
            [f for f in feature_contributions if f["contribution"] < 0],
            key=lambda x: x["contribution"]
        )

        total_spam_push = sum(f["contribution"] for f in spam_features)
        total_ham_push = sum(f["contribution"] for f in ham_features)
        net_decision_score = total_spam_push + total_ham_push + self._intercept

        # Generate interactive HTML word highlighting
        annotated_html = self._generate_annotated_html(text, token_score_lookup)

        # Generate conversational natural explanation
        explanation_text = self._build_natural_explanation(
            spam_features[:3],
            ham_features[:3],
            net_decision_score
        )

        return {
            "success": True,
            "intercept": round(self._intercept, 4),
            "total_spam_contribution": round(total_spam_push, 4),
            "total_ham_contribution": round(total_ham_push, 4),
            "net_decision_score": round(net_decision_score, 4),
            "top_spam_features": spam_features[:max_features],
            "top_ham_features": ham_features[:max_features],
            "annotated_html": annotated_html,
            "natural_explanation": explanation_text
        }

    def _generate_annotated_html(self, raw_text: str, token_scores: Dict[str, float]) -> str:
        """
        Walks through the raw text and wraps significant words in span tags
        while safely escaping HTML and preserving spaces and line breaks.
        """
        # Split text into tokens and delimiters while preserving all whitespace
        tokens_and_delims = re.split(r"(\b[a-zA-Z]+\b|[$£€¥₹]|https?://\S+|www\.\S+)", raw_text)
        output_parts: List[str] = []

        for part in tokens_and_delims:
            if not part:
                continue

            escaped = html.escape(part)
            
            # Check for currency symbols
            if part in "$£€¥₹":
                score = token_scores.get("moneysym", 0.35)
                output_parts.append(
                    f'<span class="hl-token hl-spam" data-score="{score:+.2f}" '
                    f'title="Currency Symbol: pushes +{score:.2f} toward Spam">{escaped}</span>'
                )
                continue

            # Check for URLs
            if part.startswith("http://") or part.startswith("https://") or part.startswith("www."):
                score = token_scores.get("httpaddr", 0.40)
                output_parts.append(
                    f'<span class="hl-token hl-spam" data-score="{score:+.2f}" '
                    f'title="Web Link (httpaddr): pushes +{score:.2f} toward Spam">{escaped}</span>'
                )
                continue

            # Check for alphabetic words
            if re.match(r"^[a-zA-Z]+$", part):
                stem = self._stemmer.stem(part.lower())
                score = token_scores.get(stem, 0.0)

                if score > 0.05:
                    # Positive contribution: Spam indicator
                    intensity = "high" if score > 0.25 else ("med" if score > 0.12 else "low")
                    output_parts.append(
                        f'<span class="hl-token hl-spam hl-{intensity}" data-score="{score:+.2f}" '
                        f'title="Spam Signal \'{stem}\': +{score:.2f} impact">{escaped}</span>'
                    )
                elif score < -0.05:
                    # Negative contribution: Legitimate anchor
                    intensity = "high" if score < -0.25 else ("med" if score < -0.12 else "low")
                    output_parts.append(
                        f'<span class="hl-token hl-ham hl-{intensity}" data-score="{score:+.2f}" '
                        f'title="Legitimate Anchor \'{stem}\': {score:.2f} impact">{escaped}</span>'
                    )
                else:
                    output_parts.append(escaped)
            else:
                # Delimiters, numbers, newlines
                output_parts.append(escaped.replace("\n", "<br>"))

        return "".join(output_parts)

    def _build_natural_explanation(
        self,
        top_spam: List[Dict[str, Any]],
        top_ham: List[Dict[str, Any]],
        net_score: float
    ) -> str:
        """
        Constructs an intuitive summary of the dominant drivers.
        """
        verdict = "Spam" if net_score > 0 else "Legitimate (Ham)"
        explanation_lines = [f"The model classified this email as **{verdict}**."]

        if top_spam:
            spam_terms = ", ".join([f"'{f['feature']}' (+{f['contribution']:.2f})" for f in top_spam])
            explanation_lines.append(f"Primary threat drivers: {spam_terms}.")
        
        if top_ham:
            ham_terms = ", ".join([f"'{f['feature']}' ({f['contribution']:.2f})" for f in top_ham])
            explanation_lines.append(f"Legitimate/neutral anchors: {ham_terms}.")

        if not top_spam and not top_ham:
            explanation_lines.append("The message contained predominantly neutral or unindexed vocabulary.")

        return " ".join(explanation_lines)


# Module-level convenience function
_explainer_instance = None

def explain_email(text: str, max_features: int = 8) -> Dict[str, Any]:
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = EmailExplainerService()
    return _explainer_instance.explain(text, max_features=max_features)


if __name__ == "__main__":
    sample = (
        "URGENT: You won a $1,000 Walmart Gift Card! Click https://claim-prize.net "
        "now to claim your free reward before it expires."
    )
    res = explain_email(sample)
    print("=" * 65)
    print("EXPLAINABILITY MODULE VERIFICATION")
    print("=" * 65)
    print(f"Net Decision Score: {res['net_decision_score']}")
    print(f"Natural Explanation: {res['natural_explanation']}\n")
    print("Top Spam Features:")
    for f in res["top_spam_features"]:
        print(f"  + {f['feature']:<15} | TF-IDF={f['tfidf']:.3f} | Weight={f['weight']:+.3f} | Impact={f['contribution']:+.4f}")
    print("\nAnnotated HTML preview (first 200 chars):")
    print(res["annotated_html"][:200] + "...")
