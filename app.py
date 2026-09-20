import pandas as pd
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent

MAX_FEATURES = 3000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Download Stopwords
nltk.download("stopwords")
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
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
    return vectorizer, model, metrics


vectorizer, model, metrics = train_model()
print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(metrics["report"])

def predict_email(email_text: str) -> str:
    """Classify a single email message as 'Spam' or 'Not Spam'."""
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"

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

user_input()
# # Example
# email = "Congratulations! You've won a free iPhone. Click here to claim now."
# print(f"Email: {email}\nPrediction: {predict_email(email)}")







