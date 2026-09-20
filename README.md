# Spam Email Detector

A lightweight spam email classifier built with TF-IDF and Logistic Regression.

## Features

- Preprocesses raw email text (stopword removal, stemming, lowercasing)
- Vectorizes messages with TF-IDF and trains a Logistic Regression model
- Interactive CLI: paste an email, type `END` on its own line to classify, or `exit` to quit
- Reports accuracy and a full classification report after training

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Dataset

Uses `mail_data.csv` (columns: `Category`, `Message`), loaded relative to the
script directory.