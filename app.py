import pandas as pd
import re
import argparse
import nltk
import joblib
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "artifacts"

MAX_FEATURES = 3000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Download Stopwords
nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Clean raw email text: normalize, lowercase, remove stopwords, and stem."""
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem words
    return " ".join(words)


def train_model(csv_path: Path = BASE_DIR / "mail_data.csv"):
    """Load the dataset and return a fitted (vectorizer, model, metrics) tuple."""
    df = pd.read_csv(csv_path, encoding="latin-1")[["Category", "Message"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df["cleaned_message"] = df["message"].apply(preprocess_text)

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(df["cleaned_message"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "cv_roc_auc": cv_scores.mean(),
        "report": classification_report(y_test, y_pred),
    }
    return vectorizer, model, metrics


vectorizer, model, metrics = train_model()


def save_model(path: Path = MODEL_DIR / "spam_model.joblib") -> None:
    """Persist the fitted vectorizer and model to disk with joblib."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": model}, path)
    print(f"Model saved to {path}")


def load_model(path: Path = MODEL_DIR / "spam_model.joblib"):
    """Load a previously saved vectorizer/model bundle from disk."""
    return joblib.load(path)


def predict_email(email_text: str) -> str:
    """Classify a single email message as 'Spam' or 'Not Spam'."""
    if not email_text or not email_text.strip():
        return "Not Spam (empty message)"
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    confidence = max(model.predict_proba(vectorized_text)[0])
    label = "Spam" if prediction[0] == 1 else "Not Spam"
    return f"{label} (confidence: {confidence * 100:.1f}%)"


def predict_emails(emails: list[str]) -> list[str]:
    """Classify multiple emails at once, returning one prediction per input."""
    return [predict_email(email) for email in emails]


def user_input() -> None:
    """Interactive CLI loop that classifies pasted emails until the user exits."""
    print("Paste the email (multi-line is fine). On its own line type END to classify, or 'exit' to quit :")
    while True:
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "exit":
                print("Goodbye!")
                return
            if line.strip().upper() == "END":
                break
            lines.append(line)
        email_text = "\n".join(lines)
        prediction = predict_email(email_text)
        print(f"Prediction: {prediction}")
        print()

def main() -> None:
    """Report training metrics and launch the interactive classifier."""
    parser = argparse.ArgumentParser(description="Spam email classifier")
    parser.add_argument("--demo", action="store_true", help="Classify a hardcoded sample email and exit")
    args = parser.parse_args()

    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | CV ROC-AUC: {metrics['cv_roc_auc']:.4f}")
    print(metrics["report"])

    if args.demo:
        print(predict_email("Congratulations! You've won a free iPhone. Click here to claim now."))
        return
    user_input()

if __name__ == "__main__":
    main()







