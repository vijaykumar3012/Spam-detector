import json
import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from preprocess import clean_text, phishing_check

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")
HISTORY_FILE = "history.csv"


@st.cache_resource
def load_artifacts():
    return joblib.load("models/model.pkl"), joblib.load("models/tfidf.pkl")


if not os.path.exists("models/model.pkl"):
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

model, tfidf = load_artifacts()

st.title("📧 Spam Email Detector")
st.caption("Paste any email text and find out if it's Spam or Ham.")

with st.sidebar:
    st.header("Model info")
    if os.path.exists("models/metrics.json"):
        m = json.load(open("models/metrics.json"))
        best = m["best_model"]
        st.write(f"**Model:** {best}")
        for k, v in m["results"][best].items():
            st.write(f"{k.capitalize()}: {v * 100:.2f}%")

email = st.text_area("Paste email text here", height=180)

if st.button("Check"):
    if not email.strip():
        st.warning("Please enter some text.")
    else:
        vec = tfidf.transform([clean_text(email)])
        pred = int(model.predict(vec)[0])
        spam_prob = float(model.predict_proba(vec)[0][1]) * 100
        phish_score, reasons = phishing_check(email)

        if pred == 1:
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Legit Email (Ham)")
        st.progress(int(spam_prob))
        st.write(f"**Spam probability:** {spam_prob:.1f}%")

        st.subheader("🎣 Phishing check")
        if phish_score >= 50:
            st.error(f"High phishing risk ({phish_score}/100)")
        elif phish_score >= 25:
            st.warning(f"Moderate phishing risk ({phish_score}/100)")
        else:
            st.info(f"Low phishing risk ({phish_score}/100)")
        for r in reasons:
            st.write(f"- {r}")

        row = pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email": email[:200].replace("\n", " "),
            "result": "Spam" if pred else "Ham",
            "spam_prob_%": round(spam_prob, 1),
            "phishing_score": phish_score,
        }])
        row.to_csv(HISTORY_FILE, mode="a", header=not os.path.exists(HISTORY_FILE), index=False)

st.divider()
st.subheader("🕘 History")
if os.path.exists(HISTORY_FILE):
    st.dataframe(pd.read_csv(HISTORY_FILE).tail(20).iloc[::-1], use_container_width=True)
    if st.button("Clear history"):
        os.remove(HISTORY_FILE)
        st.rerun()
else:
    st.caption("No emails checked yet.")
