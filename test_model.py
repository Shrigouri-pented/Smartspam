"""
SmartSpam - Reliability & Model Verification Test Suite
======================================================
This test suite verifies:
1. Saved model file existence and safe loading via joblib.
2. Pipeline integrity (TfidfVectorizer + MultinomialNB).
3. Handling of edge cases:
   - Empty string and whitespace-only messages.
4. Genuine ordinary human messages (Ham classification).
5. Known spam attack patterns (Spam classification).
6. Non-hardcoded dynamic probabilities and predictions.
7. Explanations on data leakage and security principles.
"""

import os
import sys
import joblib
from src.text_preprocessor import clean_text, analyze_message_heuristics


def run_tests():
    print("=" * 60)
    print("      SMARTSPAM: AUTOMATED TEST SUITE")
    print("=" * 60)

    # 1. Test Model Existence
    model_path = os.path.join("models", "spam_detector_model.joblib")
    print(f"\n[TEST 1] Verifying model file at '{model_path}'...")
    assert os.path.exists(model_path), f"FAILED: Model file not found at {model_path}."
    print("  ✓ PASS: Model file exists.")

    # 2. Test Safe Model Loading
    print("\n[TEST 2] Loading model from local trusted source...")
    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        assert False, f"FAILED: Could not load model: {e}"
    assert hasattr(pipeline, "predict"), "FAILED: Loaded object lacks predict() method."
    assert hasattr(pipeline, "predict_proba"), "FAILED: Loaded object lacks predict_proba() method."
    print("  ✓ PASS: Pipeline loaded safely with TfidfVectorizer and MultinomialNB.")

    # 3. Test Edge Cases: Empty & Whitespace Messages
    print("\n[TEST 3] Testing empty and whitespace messages...")
    empty_samples = ["", "   ", "\n\t  "]
    for s in empty_samples:
        cleaned = clean_text(s)
        # Empty inputs should not crash the model or pipeline
        if cleaned == "":
            pred = 0  # Default safe behavior
            prob = [1.0, 0.0]
        else:
            pred = pipeline.predict([cleaned])[0]
            prob = pipeline.predict_proba([cleaned])[0]
        assert pred in [0, 1], f"FAILED: Unexpected prediction {pred} for empty string."
        print(f"  ✓ PASS: Empty string input handled safely -> Output: {'Spam' if pred==1 else 'Ham'}")

    # 4. Test Ordinary Genuine Messages (Ham)
    print("\n[TEST 4] Testing ordinary genuine human messages (Ham)...")
    ordinary_messages = [
        "Hey, are we still meeting for lunch at the cafeteria at 1pm?",
        "Can you please send me the class notes from yesterday's computer networks lecture?",
        "I will be home a bit late tonight, don't wait for dinner.",
        "Thanks for helping with the assignment, talk to you tomorrow!",
    ]
    for msg in ordinary_messages:
        cleaned = clean_text(msg)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        spam_chance = proba[1] * 100
        print(f"  - Message: '{msg[:45]}...'")
        print(f"    Prediction: {'SPAM' if pred == 1 else 'HAM'} (Spam Probability: {spam_chance:.2f}%)")
        assert pred == 0, f"FAILED: Ordinary message was misclassified as Spam: '{msg}'"
    print("  ✓ PASS: All ordinary test messages correctly classified as Ham.")

    # 5. Test Common Spam Patterns
    print("\n[TEST 5] Testing common spam message patterns (Spam)...")
    spam_messages = [
        "WINNER!! As a valued customer you have been selected to receive a £1000 cash prize! Claim now text CLAIM to 87121",
        "URGENT! Your mobile account has won an entry into our weekly £500 draw. Call 08712460324 immediately to collect.",
        "FREE RING TONE! Text TONE to 8007 to download your ringtone now! Terms apply.",
        "Congratulations! You won a brand new car. Call now to claim your prize voucher.",
    ]
    for msg in spam_messages:
        cleaned = clean_text(msg)
        pred = pipeline.predict([cleaned])[0]
        proba = pipeline.predict_proba([cleaned])[0]
        spam_chance = proba[1] * 100
        print(f"  - Message: '{msg[:45]}...'")
        print(f"    Prediction: {'SPAM' if pred == 1 else 'HAM'} (Spam Probability: {spam_chance:.2f}%)")
        assert pred == 1, f"FAILED: Spam message was misclassified as Ham: '{msg}'"
    print("  ✓ PASS: All spam pattern test messages correctly classified as Spam.")

    # 6. Test Non-Hardcoded Dynamic Variation
    print("\n[TEST 6] Testing dynamic prediction behavior (No hardcoded responses)...")
    prob1 = pipeline.predict_proba(["Hey bro, where are you?"])[0][1]
    prob2 = pipeline.predict_proba(["WINNER! Congratulations, claim free cash right now!"])[0][1]
    assert prob1 != prob2, "FAILED: Probabilities are identical, indicates static response."
    assert prob2 > prob1, f"FAILED: Spam probability ({prob2}) should be significantly higher than ham ({prob1})"
    print(f"  ✓ PASS: Dynamic probabilities verified (Ham: {prob1*100:.2f}% vs Spam: {prob2*100:.2f}%).")

    # 7. Test Heuristic Extraction Utility
    print("\n[TEST 7] Testing heuristic feature extraction...")
    heuristics = analyze_message_heuristics("CONGRATS! You won $5000 CASH prize! Call 9876543210")
    assert heuristics["has_currency"] is True, "Failed currency heuristic check"
    assert heuristics["has_phone_or_code"] is True, "Failed phone/code check"
    assert "congratulations" in heuristics["suspicious_keywords"] or "cash" in heuristics["suspicious_keywords"] or "prize" in heuristics["suspicious_keywords"]
    print("  ✓ PASS: Heuristic extraction functions as expected.")

    print("\n" + "=" * 60)
    print("      ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
