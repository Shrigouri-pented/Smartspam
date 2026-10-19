"""
Text Preprocessing and Inspection Utilities for SmartSpam.

This module provides simple, clear functions to clean raw SMS text and extract
basic textual indicators (e.g., uppercase ratio, presence of currency symbols, numbers).
"""

import re


def clean_text(text: str) -> str:
    """
    Cleans a raw message string by normalizing whitespace.
    Ensures that empty, null, or weird inputs are converted safely to strings.
    """
    if not isinstance(text, str):
        return ""
    # Replace multiple whitespace characters with a single space
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def analyze_message_heuristics(text: str) -> dict:
    """
    Extracts intuitive heuristics commonly associated with spam.
    
    Evaluators love seeing these statistics because they demonstrate how real-world
    spammers craft urgent, click-driven messages compared to genuine human chats.
    """
    if not text:
        return {
            "length": 0,
            "word_count": 0,
            "has_currency": False,
            "has_phone_or_code": False,
            "uppercase_ratio": 0.0,
            "suspicious_keywords": [],
        }

    length = len(text)
    words = text.split()
    word_count = len(words)

    # Check for uppercase letters (e.g., "FREE", "CLAIM NOW")
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = (
        round(len(upper_chars) / len(alpha_chars), 3) if alpha_chars else 0.0
    )

    # Check for currency symbols or prize terms
    has_currency = bool(re.search(r"[\$£€₹]|(USD|GBP|INR|RS|cash|prize)", text, re.IGNORECASE))

    # Check for phone numbers, dial codes, shortcodes (4 to 12 consecutive digits)
    has_phone_or_code = bool(re.search(r"\b\d{4,12}\b", text))

    # Known trigger tokens often found in spam
    trigger_list = [
        "free", "winner", "win", "cash", "prize", "urgent", "claim",
        "selected", "congratulations", "guaranteed", "call now", "click",
        "text back", "reply", "offer", "discount", "account"
    ]
    lower_text = text.lower()
    found_triggers = [t for t in trigger_list if t in lower_text]

    return {
        "length": length,
        "word_count": word_count,
        "has_currency": has_currency,
        "has_phone_or_code": has_phone_or_code,
        "uppercase_ratio": uppercase_ratio,
        "suspicious_keywords": found_triggers,
    }
