import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download Stopwords
nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load dataset
df = pd.read_csv("d:\\Python Projects 2\\Spam_Email_Detector\\mail_data.csv", encoding="latin-1")[["Category", "Message"]]
df.columns = ["label", "message"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})


def preprocess_text(text):
    text = re.sub(r"\W", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]  # Remove stopwords and stem words
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["cleaned_message"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

def predict_email(email_text):
    processed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Not Spam"

def user_input():
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







