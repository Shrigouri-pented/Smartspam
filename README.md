# 🛡️ SmartSpam: Machine Learning SMS Spam Detector

A clean, reliable, and interpretable Machine Learning application designed for SMS spam classification using **Term Frequency - Inverse Document Frequency (TF-IDF)** and **Multinomial Naive Bayes**.

---

## 📁 1. Project Structure

```text
SmartSpam/
├── data/
│   └── spam.csv                     # Cleaned SMS Spam Collection dataset (5,160 unique messages)
├── models/
│   ├── spam_detector_model.joblib   # Serialized Scikit-Learn Pipeline (TF-IDF + MultinomialNB)
│   └── model_metrics.json           # Evaluation metrics & dataset split statistics
├── src/
│   ├── __init__.py                  # Package marker
│   └── text_preprocessor.py         # Text cleaning and heuristic analysis utilities
├── app.py                           # Streamlit web application with UI, inspection, & viva tabs
├── train_model.py                   # Automated data preprocessing, training, and evaluation script
├── test_model.py                    # Automated reliability and edge-case test suite
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Ignored files (virtual environments, caches)
└── README.md                        # Documentation and college viva guide
```

---

## 🎯 2. File-by-File Explanation

| File | Purpose | Key Code Blocks & Highlights |
| :--- | :--- | :--- |
| **`train_model.py`** | Trains the ML pipeline from scratch and exports evaluation metrics. | 1. `load_and_preprocess_data()`: Removes duplicate messages to prevent over-optimistic test scores.<br>2. `train_and_evaluate()`: Uses `Pipeline([('tfidf', ...), ('nb', ...)])` with `alpha=0.2` and stratified splitting.<br>3. `save_artifacts()`: Safely stores the model and metrics JSON. |
| **`app.py`** | Interactive Streamlit user interface. | 1. `load_model_pipeline()`: Safely loads the local `.joblib` model using `@st.cache_resource`.<br>2. Real-time probability prediction and text heuristic badges.<br>3. **Model & Architecture Inspector** tab reading `model_metrics.json`.<br>4. **Beginner ML & Viva Guide** tab. |
| **`test_model.py`** | Standalone automated verification suite. | Runs 7 verification tests checking model existence, safe loading, empty strings, genuine human messages, common spam patterns, dynamic probabilities, and heuristics. |
| **`src/text_preprocessor.py`** | Reusable text utilities. | 1. `clean_text()`: Strips extra whitespaces safely.<br>2. `analyze_message_heuristics()`: Scans uppercase character ratios, currency symbols, dial codes, and spam trigger keywords for user transparency. |
| **`data/spam.csv`** | Cleaned benchmark dataset. | 5,160 deduplicated messages derived from the benchmark UCI SMS Spam Collection. |
| **`models/`** | Model artifacts. | Houses the trained model binary (`.joblib`) and JSON performance report. |

---

## 🔄 3. How the Dataset Moves Through the Project

```text
[Raw SMS Dataset: 5,574 rows]
           │
           ▼
[Data Cleaning in train_model.py]
  - Drops null/empty rows
  - Identifies & removes 414 duplicate messages
  - Standardizes labels (ham -> 0, spam -> 1)
           │
           ▼
[Stratified 80/20 Train-Test Split]
  - 4,128 Training samples
  - 1,032 Held-out Testing samples
           │
           ▼
[Scikit-Learn Pipeline Fitting]
  - TF-IDF Vectorizer learns vocabulary ONLY on X_train (Prevents Data Leakage)
  - Multinomial Naive Bayes trains word-likelihood probabilities
           │
           ▼
[Evaluation & Artifact Export]
  - Evaluated on unseen X_test -> Accuracy: 98.35%, Precision: 98.26%, Recall: 88.28%
  - Saved to models/spam_detector_model.joblib & models/model_metrics.json
           │
           ▼
[Runtime Inference in app.py & test_model.py]
  - User types message -> text_preprocessor cleans text
  - Pipeline transforms text through fitted TF-IDF weights -> MultinomialNB outputs probabilities
```

---

## 🧠 4. Machine Learning Concepts in Simple Terms

### What is TF-IDF (Term Frequency - Inverse Document Frequency)?
Machine learning models cannot read raw English words directly; they require numbers.
1. **Term Frequency (TF):** How often a word appears in a specific message.
2. **Inverse Document Frequency (IDF):** How unique that word is across all messages.
   - Extremely common words like *"the"*, *"is"*, *"and"* appear in nearly all messages, so their IDF score is low.
   - Distinctive words like *"jackpot"*, *"winner"*, *"claim"*, *"urgent"* appear frequently in spam but rarely in genuine personal chat, so their IDF score is high.
3. **TF-IDF:** Multiplies TF by IDF to give each word an importance score.

### What is Multinomial Naive Bayes?
Multinomial Naive Bayes is a statistical classifier based on **Bayes' Theorem**:
$$P(\text{Spam} \mid \text{Words}) = \frac{P(\text{Words} \mid \text{Spam}) \times P(\text{Spam})}{P(\text{Words})}$$

- **Why "Naive"?** It makes the simplifying assumption that all words in a message appear independently of each other. Although words in human speech aren't strictly independent, this assumption works remarkably well, requires minimal CPU, and prevents overfitting on small datasets.
- **Why "Multinomial"?** It is designed for discrete feature counts (such as word counts or TF-IDF weights) rather than continuous bell-curve features.
- **Laplace Smoothing ($\alpha = 0.2$):** If a test message contains a word that was never seen in spam during training, its probability would be $0$. Multiplying by $0$ would ruin the entire calculation. Laplace smoothing adds a small positive constant ($\alpha$) to every word count so no word has a zero probability.

---

## ⚖️ 5. Critical Engineering & Reliability Principles

### A. Preventing Data Leakage
- **The Problem:** If you fit a TF-IDF vectorizer on your entire dataset before splitting into train and test sets, words and document frequencies from your test set leak into your training vocabulary.
- **Our Solution:** We use `sklearn.pipeline.Pipeline`. The vectorizer is fit strictly on `X_train`. When `pipeline.predict()` is called on test samples, it transforms them using only the training vocabulary.

### B. Why Duplicate Messages Cause Overly Optimistic Results
- **The Problem:** Spammers repeat identical broadcast messages (e.g., *"WIN £1000 CALL 87121"*). If 50 copies of this message exist in the raw dataset, random splitting places identical copies into both the training split and the testing split.
- **The Consequence:** The model gets tested on messages it literally memorized during training, artificially inflating accuracy to ~99.9% and hiding whether the model can generalize to new messages.
- **Our Solution:** `train_model.py` detects and removes 414 duplicate messages, ensuring evaluation reflects genuine generalization.

### C. Never Guarantee That a Predicted "Ham" Message is Safe
- **The Security Reality:** Machine learning classifiers output statistical correlations based on past training data.
- **The Risk:** An attacker can craft a targeted spear-phishing message using polite, natural language with zero typical spam keywords.
- **Our Policy:** The application labels legitimate messages as **"Ham (Statistical Estimate)"** and explicitly warns users never to treat ham predictions as cryptographic proof of safety.

---

## 🚀 6. Installation & Quickstart

### Step 1: Set up the Python Environment
```bash
# Navigate to the project directory
cd SmartSpam

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```
*Outputs: Generates `models/spam_detector_model.joblib` and `models/model_metrics.json`.*

### Step 3: Run the Reliability Test Suite
```bash
python test_model.py
```
*Outputs: Runs 7 automated checks covering model existence, safe loading, empty strings, genuine human messages, spam attacks, dynamic probabilities, and heuristics.*

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Open your browser and navigate to `http://localhost:8501`.*

---

## 🎓 7. College Project Evaluator Q&A (Viva Preparation)

### Q1: What problem does your project solve?
> **Answer:** It automatically classifies short text messages (SMS) into legitimate interpersonal communications ("Ham") or unsolicited/fraudulent messages ("Spam") using statistical Natural Language Processing, running 100% locally with high precision.

### Q2: Why did you choose Multinomial Naive Bayes instead of Deep Learning / LLMs?
> **Answer:** SMS spam classification is a well-bounded text classification problem. Multinomial Naive Bayes trains in under 2 seconds, requires zero GPU infrastructure, has near-instant inference (<5ms per message), uses minimal RAM, and achieves >98% accuracy on benchmark SMS data without the latency, financial cost, or hallucination risks of LLMs.

### Q3: What is the difference between Precision and Recall in spam detection, and which is more important?
> **Answer:**
> - **Precision:** Out of all messages our model labeled as Spam, what fraction was actually spam?
> - **Recall:** Out of all actual spam messages in the dataset, what fraction did our model catch?
> - **Which is more critical?** **Precision** is more critical. A False Positive (flagging a job offer, OTP, or hospital message as spam) can have catastrophic consequences for a user. A False Negative (one junk promotion landing in the inbox) is only a minor annoyance. Our model achieves **98.26% Precision** (only 2 false positives out of 904 test ham messages).

### Q4: What is an F1-Score?
> **Answer:** F1-score is the harmonic mean of Precision and Recall:
> $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
> It balances both false positives and false negatives, especially useful on imbalanced datasets where spam is only ~13% of all messages.

### Q5: How did you handle class imbalance in your dataset?
> **Answer:** The dataset has ~87.6% Ham and ~12.4% Spam. We handled this by:
> 1. Using **Stratified Splitting** (`stratify=y`), which ensures train and test sets have the exact same 87.6/12.4 class ratio.
> 2. Evaluating on **Precision, Recall, and F1-Score** rather than relying solely on overall accuracy.

### Q6: What is Laplace Smoothing and why is alpha=0.2 used?
> **Answer:** If a user enters a word not present in the training set vocabulary, its conditional probability would be zero. Because Naive Bayes multiplies word probabilities together, one zero would cause the whole prediction to become zero. Laplace smoothing adds $\alpha$ to word counts to prevent this zero-probability trap.

### Q7: Why do we use Scikit-Learn's Pipeline?
> **Answer:** The `Pipeline` bundles the TF-IDF vectorizer and the Naive Bayes classifier into a single executable object. It guarantees that transformations applied during training are identically applied during testing and live inference, eliminating data leakage.

### Q8: What happens when an empty or whitespace message is entered?
> **Answer:** The preprocessor checks string length after whitespace trimming. If empty, the application catches it gracefully without crashing, flags an informative alert to the user, and defaults safely.

---

## 📋 8. Implemented Features vs. Optional Future Improvements

### ✅ Actually Implemented Features:
1. **End-to-End ML Pipeline:** TF-IDF feature extraction (unigram + bigram) and Multinomial Naive Bayes classifier.
2. **Robust Data Preprocessing:** Duplicate message detection and removal to prevent evaluation bias.
3. **Stratified Train/Test Evaluation:** Verified with Accuracy (98.35%), Precision (98.26%), Recall (88.28%), F1-Score (93.00%), and Confusion Matrix.
4. **Safe Model Serialization:** Stored locally in `models/spam_detector_model.joblib`.
5. **Interactive Streamlit Web Interface:** Live classification with sample selector, dynamic probability bars, and heuristic analysis.
6. **Model & Architecture Inspector:** In-app inspection of confusion matrix and dataset breakdown.
7. **Comprehensive Viva Guide:** Embedded directly into the app and documentation.
8. **Automated Test Suite:** `test_model.py` covering edge cases, ham messages, and spam patterns.

### 🔮 Optional Future Improvements (Clear Roadmap):
1. **Active Phishing URL Reputation Lookup:** Integrate Google Safe Browsing API to detect suspicious links inside messages.
2. **Multilingual Support:** Support non-English SMS spam (e.g., Hindi, Spanish) using multilingual sentence transformers.
3. **Active Learning Feedback Loop:** Allow users to flag false positives and false negatives to incrementally retrain the local model.
4. **Offline Mobile Client:** Export the lightweight model to ONNX or TensorFlow Lite for on-device SMS filtering on Android/iOS.
