# Spam Email Detector

A lightweight spam email classifier built with TF-IDF and Logistic Regression.

## Features

- Preprocesses raw email text (stopword removal, stemming, lowercasing)
- Vectorizes messages with TF-IDF and trains a Logistic Regression model
- Interactive CLI: paste an email, type `END` on its own line to classify, or `exit` to quit
- Reports accuracy, ROC-AUC, and a full classification report after training
- Predictions include a confidence score
- Model persistence helpers (`save_model` / `load_model`)

## Setup

```bash
pip install -r requirements.txt
python app.py            # interactive mode
python app.py --demo     # quick sample classification, then exit
```

## Dataset

Uses `mail_data.csv` (columns: `Category`, `Message`), loaded relative to the
script directory.