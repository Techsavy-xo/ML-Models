"""
Composite Security Risk Scoring Engine
Author: Senior ML Engineer & Cybersecurity Architect
Project: Spam Email Classifier - Advanced Enterprise Upgrade

Calculates an interpretable Security Risk Index (0 to 100) combining
calibrated ML statistical probabilities with heuristic threat indicators.

Critical Concept:
- Model Probability: Likelihood that the text matches the spam corpus distribution.
- Security Risk Score: Operational hazard to credentials, finances, or systems.
"""

from typing import Dict, Any


def compute_risk_score(
    spam_probability: float,
    heuristic_threat_score: int,
    is_phishing_suspect: bool = False
) -> Dict[str, Any]:
    """
    Computes an interpretable, composite 0-100 cyber security risk score.

    Args:
        spam_probability: Calibrated probability P(Spam) from the ML model (0.0 to 1.0).
        heuristic_threat_score: Heuristic score from phishing analyzer (0 to 100).
        is_phishing_suspect: Whether rule-based threat engine flagged high risk.

    Returns:
        Dictionary containing risk score, severity band, factor weights, and explanation.
    """
    # Bound inputs safely
    p_spam = max(0.0, min(1.0, float(spam_probability)))
    h_score = max(0, min(100, int(heuristic_threat_score)))

    # Component weights:
    # 1. Statistical Corpus Likelihood: up to 60 points
    ml_points = p_spam * 60.0

    # 2. Heuristic Forensic Threat Signals: up to 40 points
    heuristic_points = (h_score / 100.0) * 40.0

    # Base risk score
    raw_score = ml_points + heuristic_points

    # Synergy Multiplier: If ML detects high spam probability AND phishing heuristics trigger
    if p_spam > 0.70 and h_score > 40:
        raw_score += 10.0  # Compound threat boost

    risk_score = int(round(min(100.0, max(0.0, raw_score))))

    # Determine Severity Tier
    if risk_score >= 80:
        risk_level = "CRITICAL"
        risk_color = "#ef4444"
        action_advice = "DO NOT OPEN LINKS OR ATTACHMENTS. High probability of credential harvesting or fraud."
    elif risk_score >= 55:
        risk_level = "HIGH"
        risk_color = "#f97316"
        action_advice = "Treat with extreme caution. Contains strong spam and social engineering markers."
    elif risk_score >= 30:
        risk_level = "MEDIUM"
        risk_color = "#f59e0b"
        action_advice = "Unsolicited promotional communication or ambiguous intent. Avoid sharing details."
    elif risk_score >= 15:
        risk_level = "LOW"
        risk_color = "#3b82f6"
        action_advice = "Low threat footprint, but observe standard email hygiene."
    else:
        risk_level = "MINIMAL"
        risk_color = "#10b981"
        action_advice = "Legitimate business or personal communication pattern. Minimal threat detected."

    explanation = (
        f"The Security Risk Score is {risk_score}/100 ({risk_level}). "
        f"This score combines Machine Learning statistical corpus probability "
        f"({(p_spam * 100):.1f}% contribution) with heuristic forensic indicators "
        f"({h_score}/100 threat severity). "
        "Unlike raw model probability (which simply measures lexical similarity to known spam), "
        "the Risk Score gauges potential operational harm (credential theft, financial scams, or malware)."
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "action_advice": action_advice,
        "breakdown": {
            "ml_probability_contribution": round(ml_points, 1),
            "heuristic_threat_contribution": round(heuristic_points, 1),
            "calibrated_ml_probability": round(p_spam, 4),
            "heuristic_score": h_score
        },
        "risk_explanation": explanation
    }


if __name__ == "__main__":
    test_cases = [
        (0.99, 100, True),   # Critical Phishing
        (0.95, 20, False),   # Ordinary Marketing Spam
        (0.02, 0, False),    # Normal Work Email
        (0.15, 60, True)     # Spear Phishing with subtle text but abusive links
    ]
    print("=" * 65)
    print("RISK SCORING MODULE VERIFICATION")
    print("=" * 65)
    for p, h, suspect in test_cases:
        res = compute_risk_score(p, h, suspect)
        print(f"\nP(Spam)={p:.2f}, Heuristic={h} -> Risk Score: {res['risk_score']}/100 [{res['risk_level']}]")
        print(f"Action: {res['action_advice']}")
