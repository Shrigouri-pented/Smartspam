"""
SmartSpam - Model Training Script
=================================
This script:
1. Loads the SMS spam dataset from 'data/spam.csv'.
2. Performs data inspection (checks missing values and handles duplicate rows).
3. Splits data into Train and Test sets using stratification.
4. Builds an end-to-end Machine Learning Pipeline:
   - TF-IDF Vectorizer (converts text words into statistical importance weights)
   - Multinomial Naive Bayes Classifier (calculates spam probabilities via Bayes' theorem)
5. Evaluates the model on unseen test data (Accuracy, Precision, Recall, F1-Score).
6. Saves the trained pipeline to 'models/spam_detector_model.joblib' along with
   performance metadata in 'models/model_metrics.json'.

Why this matters for College / Evaluator review:
- Pipeline prevents data leakage: TF-IDF is fit ONLY on the training split.
- Duplicate removal ensures the test set evaluates genuine generalization.
- MultinomialNB is fast, robust, and interpretable for text frequency data.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def load_and_preprocess_data(dataset_path: str = "data/spam.csv"):
    """
    Loads dataset, handles null values and duplicate records.
    
    Why duplicate removal is critical:
    If duplicate SMS messages exist and are split randomly, the exact same text
    can end up in both training and test sets. Testing the model on data it has
    already memorized leads to overly optimistic metrics. Deduplicating gives
    an honest evaluation of how the model performs on genuinely unseen messages.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Please ensure data/spam.csv exists."
        )

    print(f"[*] Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    # Validate required columns
    if "label" not in df.columns or "message" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'message' columns.")

    raw_total = len(df)
    print(f"    - Raw samples loaded: {raw_total}")

    # Drop null / empty values
    df = df.dropna(subset=["message", "label"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""]

    # Check and report duplicates
    num_duplicates = df.duplicated(subset=["message"]).sum()
    print(f"    - Found {num_duplicates} duplicate message(s).")
    
    # Remove duplicate messages for rigorous evaluation
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    clean_total = len(df)
    print(f"    - Clean unique samples: {clean_total} ({raw_total - clean_total} dropped)")

    # Standardize labels: ham -> 0, spam -> 1
    df["label_num"] = df["label"].map(lambda x: 1 if str(x).lower().strip() == "spam" else 0)
    
    ham_count = (df["label_num"] == 0).sum()
    spam_count = (df["label_num"] == 1).sum()
    print(f"    - Class distribution: Ham = {ham_count} ({ham_count/clean_total*100:.1f}%), Spam = {spam_count} ({spam_count/clean_total*100:.1f}%)")

    return df, {
        "raw_samples": int(raw_total),
        "clean_samples": int(clean_total),
        "duplicates_removed": int(num_duplicates),
        "ham_count": int(ham_count),
        "spam_count": int(spam_count),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Splits data, trains the TF-IDF + MultinomialNB pipeline, and calculates metrics.
    
    Why scikit-learn Pipeline?
    The Pipeline encapsulates both text vectorization and classification.
    When we call pipeline.fit(X_train, y_train), the TF-IDF vocabulary is learned
    EXCLUSIVELY from X_train. X_test is never seen during vocabulary creation,
    completely preventing data leakage.
    """
    X = df["message"]
    y = df["label_num"]

    # Stratified split ensures equal class proportions in train and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n[*] Dataset Split (test_size={test_size}):")
    print(f"    - Training samples: {len(X_train)} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f"    - Testing samples:  {len(X_test)} (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})")

    # Construct the Machine Learning Pipeline
    # 1. TfidfVectorizer: converts raw text into numerical n-gram importance
    # 2. MultinomialNB: probabilistic classifier with Laplace smoothing (alpha=0.2)
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),  # captures single words and pairs like 'cash prize', 'call now'
                max_features=5000,
                sublinear_tf=True,   # dampens the effect of repetitive words
            ),
        ),
        (
            "nb",
            MultinomialNB(alpha=0.2),  # Laplace smoothing parameter
        ),
    ])

    print("\n[*] Training Multinomial Naive Bayes model...")
    pipeline.fit(X_train, y_train)
    print("    - Model training completed successfully.")

    # Evaluate on the held-out test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 50)
    print("           MODEL EVALUATION ON TEST SET")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}% (How many predicted spams were truly spam)")
    print(f"Recall:    {recall * 100:.2f}% (How many real spams were caught)")
    print(f"F1-Score:  {f1 * 100:.2f}% (Harmonic mean of precision & recall)")
    print("\nConfusion Matrix:")
    print(f"  True Ham  [TN: {cm[0][0]}, FP: {cm[0][1]}]")
    print(f"  True Spam [FN: {cm[1][0]}, TP: {cm[1][1]}]")
    print("=" * 50)

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1],
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "algorithm": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF (unigrams + bigrams, sublinear_tf)",
    }

    return pipeline, metrics


def save_artifacts(pipeline, metrics: dict, dataset_stats: dict, output_dir: str = "models"):
    """
    Saves the trained pipeline and metadata safely to the models/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, "spam_detector_model.joblib")
    metrics_file = os.path.join(output_dir, "model_metrics.json")

    print(f"\n[*] Saving model pipeline to '{model_file}'...")
    joblib.dump(pipeline, model_file)

    combined_metadata = {
        **metrics,
        "dataset_stats": dataset_stats,
    }

    print(f"[*] Saving evaluation metadata to '{metrics_file}'...")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(combined_metadata, f, indent=2)

    print("[+] All artifacts saved successfully.")


def main():
    print("=== SmartSpam: Model Training & Evaluation Pipeline ===")
    df, stats = load_and_preprocess_data("data/spam.csv")
    pipeline, metrics = train_and_evaluate(df)
    save_artifacts(pipeline, metrics, stats)
    print("\n[✓] Training pipeline finished. Ready for app and testing!")


if __name__ == "__main__":
    main()
