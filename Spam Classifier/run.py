"""
Spam Email Classifier - Application Entry Point
Author: Senior Full-Stack ML Engineer
Project: End-to-End Spam Classification System

Starts the production-ready Flask server and exposes the modern SaaS dashboard.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import app
from app.database import init_db

def main():
    """
    Bootstraps the database and starts the local web server.
    """
    model_path = os.path.join(PROJECT_ROOT, "models", "spam_model.joblib")
    vectorizer_path = os.path.join(PROJECT_ROOT, "models", "tfidf_vectorizer.joblib")

    # If models are not yet trained, prompt or auto-train
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("\n[!] Trained model artifacts not detected. Launching ML training pipeline...")
        from src.train import train_and_evaluate_all
        train_and_evaluate_all()

    # Initialize SQLite database schema
    init_db()

    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")

    print("\n" + "=" * 65)
    print("   SENTINELSPAM AI - THREAT INTELLIGENCE APPLICATION")
    print("=" * 65)
    print(f" * Server running on: http://{host}:{port}")
    print(f" * REST API Documentation: http://{host}:{port}/api/stats")
    print(" * Press CTRL+C to stop the server.")
    print("=" * 65 + "\n")

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
