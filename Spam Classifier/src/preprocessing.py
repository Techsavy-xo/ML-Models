"""
Text Preprocessing & NLP Normalization Module
Author: Senior ML Engineer
Project: Spam Email Classifier - End-to-End Machine Learning System

This module handles:
1. Low-level text sanitization (HTML, URLs, email patterns).
2. Punctuation stripping and case normalization.
3. Word tokenization using NLTK (with safe fallback).
4. English stopword filtering.
5. Morphological stemming using PorterStemmer.
"""

import re
import logging
from typing import List, Union
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Ensure NLTK dependencies are available
def _ensure_nltk_resources():
    required_packages = ["stopwords", "punkt", "punkt_tab"]
    for pkg in required_packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                logger.warning(f"Could not automatically download NLTK package {pkg}: {e}")

_ensure_nltk_resources()

# Initialize stemmer and stopword vocabulary
try:
    _STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # Minimal fallback stopword set in case of offline environment issues
    _STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
        "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
        "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
        "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
        "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
        "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
        "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
        "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
        "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
        "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
        "your", "yours", "yourself", "yourselves"
    }

_STEMMER = PorterStemmer()

# Regex patterns precompiled for performance
_HTML_PATTERN = re.compile(r"<[^>]+>")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
_CURRENCY_PATTERN = re.compile(r"[$£€¥₹]")
_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: Union[str, float, None]) -> str:
    """
    Cleans and normalizes email/message text for NLP feature extraction.

    Pipeline operations:
    1. Null & type check.
    2. Lowercase transformation.
    3. Removal of HTML markup.
    4. Canonical token replacement for URLs (httpaddr), emails (emailaddr), and currencies (moneysym).
    5. Removal of non-alphabetic punctuation and noise.
    6. Tokenization.
    7. Stopword filtering.
    8. Porter stemming.

    Args:
        text (str | None): Raw email or SMS string.

    Returns:
        str: Space-separated preprocessed tokens.
    """
    if text is None or not isinstance(text, str):
        return ""

    # 1. Lowercase
    normalized = text.lower().strip()
    if not normalized:
        return ""

    # 2. Strip HTML tags
    normalized = _HTML_PATTERN.sub(" ", normalized)

    # 3. Canonical token substitution for high-signal email structures
    normalized = _URL_PATTERN.sub(" httpaddr ", normalized)
    normalized = _EMAIL_PATTERN.sub(" emailaddr ", normalized)
    normalized = _CURRENCY_PATTERN.sub(" moneysym ", normalized)

    # 4. Remove special characters and numbers (retain purely alphabet tokens)
    normalized = _NON_ALPHA_PATTERN.sub(" ", normalized)

    # 5. Tokenize
    try:
        tokens = word_tokenize(normalized)
    except Exception:
        # Fallback regex tokenization if NLTK tokenizer fails
        tokens = normalized.split()

    # 6. Stopword filtering and Stemming
    cleaned_tokens: List[str] = [
        _STEMMER.stem(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 1
    ]

    return " ".join(cleaned_tokens)


def preprocess_series(series) -> List[str]:
    """
    Preprocesses an entire Pandas Series or iterable of texts.
    """
    return [clean_text(item) for item in series]


if __name__ == "__main__":
    sample_emails = [
        "Congratulations! You won a $1,000 Walmart Gift Card. Click https://spam.link to claim now!",
        "Subject: Team Sync Tomorrow at 10 AM. Please review the attached slide deck before the call.",
        "URGENT: Your bank account has been locked. Verify identity at http://secure-login.phishing.net",
        "Hi Mom, I will be home by 7 PM. Let me know if you need any groceries."
    ]
    
    print("=" * 60)
    print("PREPROCESSING MODULE VERIFICATION")
    print("=" * 60)
    for email in sample_emails:
        print(f"\n[RAW]:     {email}")
        print(f"[CLEANED]: {clean_text(email)}")
