"""
Automated Test Suite - Enterprise Threat Intelligence & Explainable AI
Author: Senior ML Engineer & Full-Stack Developer
Project: Spam Email Classifier - Advanced Enterprise Upgrade

Comprehensive test suite covering:
1. NLP preprocessing and token normalization.
2. Machine learning inference accuracy and schema adherence.
3. Explainable AI (XAI) feature attributions and HTML span highlights.
4. Heuristic phishing analysis (URLs, urgency, credential harvesting, brand spoofing).
5. Composite 0-100 cyber security risk scoring.
6. High-throughput bulk CSV classification and report export.
7. SQLite persistence, metrics aggregation, and history deletion.
8. Flask REST API endpoints and error validation.
"""

import os
import sys
import json
import io
import pytest
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import clean_text
from src.predict import predict_email
from src.explainability import explain_email
from src.phishing_analyzer import analyze_phishing_threats
from src.risk_scoring import compute_risk_score
from src.bulk_processor import process_bulk_dataframe, generate_export_csv, detect_email_column
from app.database import (
    init_db,
    save_prediction,
    get_history,
    clear_history,
    get_classification_stats
)
from app.app import app


# ---------------------------------------------------------------------------
# 1. NLP Preprocessing Tests
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_clean_text_lowercasing_and_punctuation(self):
        raw = "HELLO WORLD! This is an NLP Test... Let's classify."
        cleaned = clean_text(raw)
        assert "hello" in cleaned
        assert "world" in cleaned
        assert "!" not in cleaned
        assert "." not in cleaned

    def test_clean_text_replaces_urls_and_emails(self):
        raw = "Visit http://spam-site.com or email admin@scam.org for prizes!"
        cleaned = clean_text(raw)
        assert "httpaddr" in cleaned
        assert "emailaddr" in cleaned
        assert "http://spam-site.com" not in cleaned
        assert "admin@scam.org" not in cleaned

    def test_clean_text_normalizes_currency_symbols(self):
        raw = "Win $500 or £1000 cash today!"
        cleaned = clean_text(raw)
        assert "moneysym" in cleaned

    def test_clean_text_handles_empty_and_none(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text("     ") == ""
        assert clean_text("12345!@#%^") == ""


# ---------------------------------------------------------------------------
# 2. Machine Learning Inference Pipeline Tests
# ---------------------------------------------------------------------------
class TestMLInference:
    def test_artifacts_exist(self):
        model_path = os.path.join(PROJECT_ROOT, "models", "spam_model.joblib")
        vec_path = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.joblib")
        assert os.path.exists(model_path), "Champion model file missing!"
        assert os.path.exists(vec_path), "TF-IDF vectorizer file missing!"

    def test_predict_email_spam_classification(self):
        spam_msg = (
            "URGENT! You have won a £1,000,000 cash prize in our mobile lottery. "
            "Call 0800123456 now or text CLAIM to 88010 to receive your cheque."
        )
        res = predict_email(spam_msg)
        assert res["success"] is True
        assert res["is_spam"] is True
        assert res["label"] == "Spam"
        assert res["confidence"] >= 50.0
        assert res["probabilities"]["spam"] > 0.5

    def test_predict_email_ham_classification(self):
        ham_msg = (
            "Hi David, attached is the revised financial report and slide deck for "
            "tomorrow's management meeting at 10 AM. Please review and let me know."
        )
        res = predict_email(ham_msg)
        assert res["success"] is True
        assert res["is_spam"] is False
        assert res["label"] == "Ham"
        assert res["confidence"] >= 50.0
        assert res["probabilities"]["ham"] > 0.5

    def test_predict_email_schema(self):
        msg = "Hey, are we still meeting for lunch today?"
        res = predict_email(msg)
        required_keys = [
            "success", "is_spam", "label", "confidence", "confidence_score",
            "probabilities", "cleaned_tokens", "char_count", "word_count", "preview"
        ]
        for key in required_keys:
            assert key in res, f"Expected key '{key}' missing from prediction response."

    def test_predict_email_empty_string(self):
        res = predict_email("")
        assert res["success"] is False
        assert "error" in res


# ---------------------------------------------------------------------------
# 3. Explainable AI (XAI) Tests
# ---------------------------------------------------------------------------
class TestExplainability:
    def test_explain_email_returns_attributions(self):
        sample = "URGENT: You won a $1000 cash prize! Call now to claim free gift."
        exp = explain_email(sample)
        assert exp["success"] is True
        assert "top_spam_features" in exp
        assert "top_ham_features" in exp
        assert "annotated_html" in exp
        assert "natural_explanation" in exp
        assert "hl-token" in exp["annotated_html"]
        assert len(exp["top_spam_features"]) > 0

    def test_explain_email_empty_input(self):
        exp = explain_email("")
        assert exp["success"] is False
        assert "error" in exp


# ---------------------------------------------------------------------------
# 4. Phishing & Threat Forensics Tests
# ---------------------------------------------------------------------------
class TestPhishingAnalyzer:
    def test_phishing_analyzer_detects_ip_and_urgency(self):
        text = (
            "URGENT: Your Microsoft 365 account has been suspended! "
            "Click http://192.168.1.1/login.php within 24 hours to verify your password."
        )
        res = analyze_phishing_threats(text)
        assert res["is_phishing_suspect"] is True
        assert res["threat_score"] >= 65
        assert res["threat_level"] in ["HIGH", "CRITICAL"]
        assert len(res["url_analysis"]["flags"]) > 0
        assert len(res["urgency_indicators"]) > 0
        assert len(res["credential_harvesting"]) > 0
        assert len(res["impersonation_indicators"]) > 0

    def test_phishing_analyzer_disclaimer_present(self):
        res = analyze_phishing_threats("Hey, what time are we meeting?")
        assert "disclaimer" in res
        assert "HEURISTIC THREAT DISCLAIMER" in res["disclaimer"]

    def test_phishing_analyzer_empty_input(self):
        res = analyze_phishing_threats("")
        assert res["threat_level"] == "NONE_DETECTED"
        assert res["threat_score"] == 0


# ---------------------------------------------------------------------------
# 5. Composite Risk Scoring Tests
# ---------------------------------------------------------------------------
class TestRiskScoring:
    def test_risk_scoring_critical_threat(self):
        res = compute_risk_score(spam_probability=0.99, heuristic_threat_score=100, is_phishing_suspect=True)
        assert res["risk_score"] >= 80
        assert res["risk_level"] == "CRITICAL"
        assert "DO NOT OPEN" in res["action_advice"]

    def test_risk_scoring_minimal_threat(self):
        res = compute_risk_score(spam_probability=0.01, heuristic_threat_score=0, is_phishing_suspect=False)
        assert res["risk_score"] < 15
        assert res["risk_level"] == "MINIMAL"

    def test_risk_scoring_bounds(self):
        res = compute_risk_score(spam_probability=1.5, heuristic_threat_score=150)
        assert 0 <= res["risk_score"] <= 100


# ---------------------------------------------------------------------------
# 6. Bulk Classification Tests
# ---------------------------------------------------------------------------
class TestBulkProcessor:
    def test_process_bulk_dataframe(self):
        df = pd.DataFrame({
            "message": [
                "URGENT: Win cash now http://spam.xyz",
                "Team sync tomorrow at 10 AM in Conference Room B."
            ]
        })
        summary, enriched = process_bulk_dataframe(df)
        assert summary["total_emails"] == 2
        assert summary["spam_count"] == 1
        assert summary["ham_count"] == 1
        assert len(enriched) == 2
        assert "risk_score" in enriched.columns

    def test_bulk_csv_export_string(self):
        df = pd.DataFrame({
            "email": ["Free prize claim now!", "Let's review the document."]
        })
        _, enriched = process_bulk_dataframe(df)
        csv_str = generate_export_csv(enriched)
        assert "id,preview,classification" in csv_str

    def test_bulk_detect_column(self):
        df = pd.DataFrame({"mail_body": ["Hello world"]})
        col = detect_email_column(df)
        assert col == "mail_body"


# ---------------------------------------------------------------------------
# 7. Database Persistence & History Tests
# ---------------------------------------------------------------------------
class TestDatabase:
    def setup_method(self):
        init_db()
        clear_history()

    def test_save_and_retrieve_history(self):
        save_prediction("Test email 1", "Spam", True, 98.5)
        save_prediction("Test email 2", "Ham", False, 99.1)
        history = get_history(limit=10)
        assert len(history) == 2
        assert history[0]["label"] == "Ham"
        assert history[1]["label"] == "Spam"

    def test_classification_stats(self):
        save_prediction("Spam 1", "Spam", True, 90.0)
        save_prediction("Spam 2", "Spam", True, 95.0)
        save_prediction("Ham 1", "Ham", False, 99.0)

        stats = get_classification_stats()
        assert stats["total_emails"] == 3
        assert stats["total_spam"] == 2
        assert stats["total_ham"] == 1
        assert stats["spam_percentage"] == pytest.approx(66.67, 0.1)

    def test_clear_history(self):
        save_prediction("Message", "Ham", False, 95.0)
        deleted = clear_history()
        assert deleted == 1
        assert len(get_history()) == 0


# ---------------------------------------------------------------------------
# 8. Flask REST API Integration Tests
# ---------------------------------------------------------------------------
class TestFlaskAPI:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client

    def test_index_page_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"SentinelSpam" in response.data

    def test_api_stats_returns_roc_auc_and_curves(self, client):
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "roc_auc" in data["model_benchmarks"]
        assert data["model_benchmarks"]["roc_auc"] > 0.95
        assert "all_models" in data["model_benchmarks"]

    def test_api_predict_returns_xai_and_phishing(self, client):
        payload = {
            "email_text": "URGENT: Microsoft account suspended. Verify at http://192.168.1.1/login"
        }
        response = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["label"] == "Spam"
        assert "explainability" in data
        assert "phishing_analysis" in data
        assert "risk_assessment" in data
        assert data["risk_assessment"]["risk_score"] > 50

    def test_api_explain_endpoint(self, client):
        payload = {"email_text": "Win cash prize today!"}
        res = client.post("/api/explain", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "annotated_html" in data

    def test_api_bulk_predict_and_download(self, client):
        csv_data = b"email\nFree cash prize call now\nMeeting at 10am tomorrow\n"
        res = client.post(
            "/api/bulk-predict",
            data={"file": (io.BytesIO(csv_data), "batch.csv")},
            content_type="multipart/form-data"
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["summary"]["total_emails"] == 2

        # Download CSV
        dl_res = client.get("/api/download-bulk")
        assert dl_res.status_code == 200
        assert "attachment" in dl_res.headers.get("Content-Disposition", "")

    def test_api_predict_empty_input_returns_400(self, client):
        payload = {"email_text": "    "}
        response = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "error" in data

    def test_api_history_lifecycle(self, client):
        client.delete("/api/history")
        client.post(
            "/api/predict",
            data=json.dumps({"email_text": "Lunch at noon?"}),
            content_type="application/json"
        )
        get_res = client.get("/api/history")
        assert get_res.status_code == 200
        hist_data = get_res.get_json()
        assert hist_data["count"] == 1
