"""
Heuristic Phishing & Cyber Threat Intelligence Analyzer
Author: Senior ML Engineer & Cybersecurity Specialist
Project: Spam Email Classifier - Advanced Enterprise Upgrade

This module implements rule-based heuristic threat detection decoupled
from the statistical ML model. It inspects:
1. Suspicious URL structures (IP hosts, abusive TLDs, shorteners, @ trickery).
2. Psychological coercion and urgency language.
3. Credential harvesting and fake authentication demands.
4. High-value brand impersonation and spoofing attempts.

Strict Safety Standard:
Never claims an email is safe simply because no heuristic patterns were detected.
"""

import re
from typing import Dict, Any, List
from urllib.parse import urlparse

# Precompiled Regex Patterns for Performance
_URL_REGEX = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_IP_IN_URL_REGEX = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$)", re.IGNORECASE)
_SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "buzz", "work", "click", "rest", "loan", "support", "live"
}
_URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly"
}

_URGENCY_PATTERNS = [
    (r"\b(?:within\s+(?:24|12|48)\s*hours?)\b", "Strict Time Deadline (12-48h Ultimatum)"),
    (r"\b(?:immediat(?:e|ely)\s+(?:action|respons|verif|protect|resolv))\b", "Immediate Action Required"),
    (r"\b(?:account\s+(?:has\s+been\s+)?(?:suspend|limit|lock|deactivat|restrict))\b", "Account Suspension / Lock Warning"),
    (r"\b(?:unauthor(?:ized)?\s+(?:access|login|attempt|activ))\b", "Unauthorized Login Claim"),
    (r"\b(?:secur(?:ity)?\s+(?:alert|breach|compromis))\b", "Security Alert Coercion"),
    (r"\b(?:final\s+(?:warn|notic|chanc))\b", "Final Warning / Ultimatum"),
    (r"\b(?:urgent(?:ly)?\s+(?:requir|action|notic))\b", "Urgent Notice"),
    (r"\b(?:fail(?:ure)?\s+to\s+compl(?:y|iance))\b", "Threat of Legal/Account Action")
]

_CREDENTIAL_HARVESTING_PATTERNS = [
    (r"\b(?:verif(?:y|ication)\s+(?:your\s+)?(?:account|ident|passcode|password|credenti))\b", "Credential Verification Demand"),
    (r"\b(?:updat(?:e)?\s+(?:your\s+)?(?:bill|payment|bank|card|detail))\b", "Financial / Billing Detail Request"),
    (r"\b(?:confirm\s+(?:your\s+)?(?:pass(?:word)?|pin|ssn|social\s+secur))\b", "Sensitive PIN/SSN Confirmation"),
    (r"\b(?:click\s+(?:here\s+)?to\s+(?:login|sign\s*in|restor|unlock))\b", "Direct Login / Unlock Call-to-Action"),
    (r"\b(?:log\s*in\s+(?:to\s+)?(?:your\s+)?(?:portal|account)\s+immediat)\b", "Urgent Portal Login Prompt"),
    (r"\b(?:provid(?:e)?\s+(?:your\s+)?(?:one-time\s+pass|otp|code))\b", "OTP / 2FA Interception Pattern")
]

_TARGETED_BRANDS = [
    ("microsoft", ["office365", "outlook", "microsoft", "teams", "onedrive"]),
    ("paypal", ["paypal", "paypal-service"]),
    ("google", ["google", "gmail", "google-drive"]),
    ("apple", ["apple", "icloud", "itunes", "appleid"]),
    ("amazon", ["amazon", "prime"]),
    ("netflix", ["netflix"]),
    ("chase", ["chase", "jpmorgan"]),
    ("wells fargo", ["wellsfargo", "wells-fargo"]),
    ("bank of america", ["bankofamerica", "bofa"]),
    ("dhl", ["dhl", "dhl-express"]),
    ("fedex", ["fedex"]),
    ("usps", ["usps", "postal-service"])
]


def analyze_phishing_threats(email_text: str) -> Dict[str, Any]:
    """
    Executes rule-based heuristic inspection of an email for phishing attack patterns.

    Args:
        email_text: Raw incoming email body and subject.

    Returns:
        Structured threat assessment dictionary containing:
        - is_phishing_suspect: bool
        - threat_level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', or 'NONE_DETECTED'
        - threat_score: int (0 to 100)
        - url_analysis: details of URLs found and flags
        - urgency_indicators: list of urgency patterns triggered
        - credential_harvesting: list of auth harvesting patterns
        - impersonation_indicators: list of brand spoofing markers
        - disclaimer: legal and security standard disclaimer
    """
    if not email_text or not isinstance(email_text, str) or not email_text.strip():
        return {
            "is_phishing_suspect": False,
            "threat_level": "NONE_DETECTED",
            "threat_score": 0,
            "url_analysis": {"urls_found": [], "flags": []},
            "urgency_indicators": [],
            "credential_harvesting": [],
            "impersonation_indicators": [],
            "total_indicators_count": 0,
            "disclaimer": "No content provided to evaluate."
        }

    text_lower = email_text.lower()
    threat_points = 0
    all_flags = []

    # 1. URL Analysis
    urls = _URL_REGEX.findall(email_text)
    url_flags = []
    
    for url in urls:
        # Check IP address as hostname
        if _IP_IN_URL_REGEX.search(url):
            url_flags.append(f"Direct IP Hostname detected: '{url}' (High-risk obfuscation)")
            threat_points += 35

        # Check @ symbol in URL (credentials trickery)
        if "@" in url:
            url_flags.append(f"Embedded '@' in URL: '{url}' (Browser redirection trick)")
            threat_points += 30

        try:
            parsed = urlparse(url if url.startswith("http") else "http://" + url)
            host = parsed.netloc.lower()
            
            # Check for URL shorteners
            if any(shortener in host for shortener in _URL_SHORTENERS):
                url_flags.append(f"URL Shortener detected: '{host}' (Masks destination domain)")
                threat_points += 20

            # Check for suspicious TLDs
            tld = host.split(".")[-1] if "." in host else ""
            if tld in _SUSPICIOUS_TLDS:
                url_flags.append(f"High-abuse top-level domain '.{tld}' in '{host}'")
                threat_points += 25

            # Excessive subdomains (e.g. login.microsoft.security.xyz.com)
            parts = host.split(".")
            if len(parts) > 4:
                url_flags.append(f"Excessive subdomains ({len(parts)} levels) in '{host}'")
                threat_points += 15

        except Exception:
            pass

    # 2. Urgency and Psychological Coercion
    urgency_flags = []
    for pattern, desc in _URGENCY_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            urgency_flags.append({"pattern": desc, "matched_text": match.group(0)})
            threat_points += 15

    # 3. Credential Harvesting & Fake Authentication
    credential_flags = []
    for pattern, desc in _CREDENTIAL_HARVESTING_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            credential_flags.append({"pattern": desc, "matched_text": match.group(0)})
            threat_points += 25

    # 4. Brand Impersonation
    impersonation_flags = []
    for brand, keywords in _TARGETED_BRANDS:
        found_brand = any(kw in text_lower for kw in keywords)
        if found_brand:
            # If brand is mentioned AND email contains URLs or credential demands
            has_urls = len(urls) > 0
            has_cred_demands = len(credential_flags) > 0 or len(urgency_flags) > 0
            
            if has_urls and has_cred_demands:
                impersonation_flags.append({
                    "brand": brand.title(),
                    "reason": f"References {brand.title()} with external links and urgent credential prompts"
                })
                threat_points += 25
            elif has_cred_demands:
                impersonation_flags.append({
                    "brand": brand.title(),
                    "reason": f"References {brand.title()} accompanied by security/authentication demands"
                })
                threat_points += 15

    # Calculate overall threat score (capped at 100)
    final_score = min(100, threat_points)
    
    if final_score >= 65:
        threat_level = "CRITICAL"
        is_suspect = True
    elif final_score >= 40:
        threat_level = "HIGH"
        is_suspect = True
    elif final_score >= 20:
        threat_level = "MEDIUM"
        is_suspect = True
    elif final_score > 0:
        threat_level = "LOW"
        is_suspect = False
    else:
        threat_level = "NONE_DETECTED"
        is_suspect = False

    total_indicators = len(url_flags) + len(urgency_flags) + len(credential_flags) + len(impersonation_flags)

    # Required Safety Standard Disclaimer
    safety_disclaimer = (
        "HEURISTIC THREAT DISCLAIMER: "
        + ("No overt heuristic phishing signatures were matched. Note that absence of rule-based triggers "
           "does NOT guarantee an email is safe. Advanced spear-phishing frequently evades static signatures. "
           "Always evaluate the calibrated Machine Learning prediction."
           if total_indicators == 0 else
           "Heuristic threat indicators are rule-based forensic observations. They operate independently "
           "from the Machine Learning model to detect high-risk phishing and coercion tactics.")
    )

    return {
        "is_phishing_suspect": is_suspect,
        "threat_level": threat_level,
        "threat_score": final_score,
        "total_indicators_count": total_indicators,
        "url_analysis": {
            "urls_detected": len(urls),
            "urls_list": urls[:5],
            "flags": url_flags
        },
        "urgency_indicators": urgency_flags,
        "credential_harvesting": credential_flags,
        "impersonation_indicators": impersonation_flags,
        "disclaimer": safety_disclaimer
    }


if __name__ == "__main__":
    test_email = (
        "URGENT: Your Microsoft 365 account has been suspended due to unauthorized access. "
        "Immediate action required! Click http://192.168.1.50/login.php within 24 hours to "
        "verify your password and prevent permanent deletion."
    )
    result = analyze_phishing_threats(test_email)
    print("=" * 65)
    print("PHISHING ANALYZER MODULE VERIFICATION")
    print("=" * 65)
    print(f"Threat Level: {result['threat_level']} (Score: {result['threat_score']}/100)")
    print(f"Is Phishing Suspect: {result['is_phishing_suspect']}")
    print(f"Total Indicators Triggered: {result['total_indicators_count']}")
    print("\nURL Flags:")
    for f in result["url_analysis"]["flags"]:
        print(f"  - {f}")
    print("\nUrgency Flags:")
    for u in result["urgency_indicators"]:
        print(f"  - {u['pattern']} -> '{u['matched_text']}'")
    print("\nCredential Flags:")
    for c in result["credential_harvesting"]:
        print(f"  - {c['pattern']} -> '{c['matched_text']}'")
    print("\nImpersonation Flags:")
    for i in result["impersonation_indicators"]:
        print(f"  - {i['brand']}: {i['reason']}")
    print(f"\nDisclaimer:\n{result['disclaimer']}")
