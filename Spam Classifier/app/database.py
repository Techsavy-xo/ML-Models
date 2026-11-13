"""
Database Management Module (SQLite)
Author: Senior Full-Stack ML Engineer
Project: Spam Email Classifier - End-to-End Machine Learning System

Manages SQLite persistence for prediction history and dynamic user metrics.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "spam_history.db")


def get_db_connection() -> sqlite3.Connection:
    """
    Creates and returns a connection to the SQLite database.
    Row factory is configured to sqlite3.Row for dict-like access.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Initializes database schema and ensures required tables exist.
    """
    logger.info(f"Initializing SQLite database at: {DB_PATH}")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email_preview TEXT NOT NULL,
                full_text TEXT NOT NULL,
                label TEXT NOT NULL,
                is_spam INTEGER NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("Database schema verified.")


def save_prediction(
    full_text: str,
    label: str,
    is_spam: bool,
    confidence: float
) -> int:
    """
    Saves a new classification record to the history table.
    """
    preview = full_text.strip()[:100] + ("..." if len(full_text.strip()) > 100 else "")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (email_preview, full_text, label, is_spam, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (preview, full_text, label, 1 if is_spam else 0, confidence, now_str))
        conn.commit()
        record_id = cursor.lastrowid
    return record_id


def get_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent prediction history entries.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, email_preview, label, is_spam, confidence, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def clear_history() -> int:
    """
    Deletes all records from the predictions table.
    Returns count of deleted records.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions'")
        conn.commit()
    return count


def get_classification_stats() -> Dict[str, Any]:
    """
    Aggregates runtime classification metrics from the database.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_count,
                COALESCE(SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END), 0) as spam_count,
                COALESCE(SUM(CASE WHEN is_spam = 0 THEN 1 ELSE 0 END), 0) as ham_count,
                COALESCE(AVG(confidence), 0.0) as avg_confidence
            FROM predictions
        """)
        row = cursor.fetchone()
        
        total = row["total_count"]
        spam = row["spam_count"]
        ham = row["ham_count"]
        avg_conf = round(float(row["avg_confidence"]), 2)
        spam_pct = round((spam / total * 100), 2) if total > 0 else 0.0
        
        return {
            "total_emails": total,
            "total_spam": spam,
            "total_ham": ham,
            "spam_percentage": spam_pct,
            "avg_confidence": avg_conf
        }


# Auto-initialize database schema when imported
init_db()
