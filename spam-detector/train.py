"""Train spam detector: python train.py [--data data/spam.csv]"""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from preprocess import clean_text


def load_data(path):
    # latin-1 handles the popular Kaggle "spam.csv" (columns v1, v2)
    df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower(): c for c in df.columns}
    text_col = next(cols[c] for c in ["text", "email text", "email", "message", "v2"] if c in cols)
    label_col = next(cols[c] for c in ["label", "category", "class", "v1"] if c in cols)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].astype(str).str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
    return df


def main(path):
    df = load_data(path)
    print("Total emails:", len(df))
    print("Missing values:\n", df.isnull().sum())
    df = df.dropna().drop_duplicates()
    print("Label counts:\n", df["label"].value_counts())
    print(f"Spam %: {df['label'].mean() * 100:.1f}%")

    df["clean"] = df["text"].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"].astype(int), test_size=0.2, random_state=42,
        stratify=df["label"])

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        p = m.predict(Xte)
        results[name] = {
            "accuracy": accuracy_score(y_test, p),
            "precision": precision_score(y_test, p, zero_division=0),
            "recall": recall_score(y_test, p, zero_division=0),
            "f1": f1_score(y_test, p, zero_division=0),
        }
        fitted[name] = (m, p)
        print(f"\n{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    best = max(results, key=lambda n: results[n]["f1"])
    print("\nBest model:", best)
    model, preds = fitted[best]

    joblib.dump(model, "models/model.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump({"best_model": best, "results": results}, f, indent=2)

    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"Confusion Matrix - {best}")
    plt.savefig("models/confusion_matrix.png", bbox_inches="tight")
    print("Saved models/model.pkl, tfidf.pkl, metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spam.csv")
    main(ap.parse_args().data)
