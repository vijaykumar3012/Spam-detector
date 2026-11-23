# 📧 Spam Email Detector (NLP)

Classifies email text as **Spam** or **Ham** using NLTK, TF-IDF and scikit-learn, with a Streamlit UI,
spam probability, rule-based phishing check and history.

## Setup
```bash
pip install -r requirements.txt
```
## Dataset
Put a CSV at `data/spam.csv`. Accepted formats (auto-detected):
- Kaggle "SMS Spam Collection": columns `v1` (ham/spam), `v2` (text)
- Or columns `text`, `label` (0 = Ham, 1 = Spam)

A tiny `data/sample_spam.csv` is included only so the code runs out of the box. Use a real dataset for 97%+ accuracy.

## Train
```bash
python train.py --data data/spam.csv
```
## Run app
```bash
streamlit run app.py
```
## Deploy on Hugging Face Spaces
1. Create a Space -> SDK: **Streamlit**.
2. Upload `app.py`, `preprocess.py`, `requirements.txt` and the `models/` folder (model.pkl, tfidf.pkl, metrics.json).
3. The Space builds automatically.
