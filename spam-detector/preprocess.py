"""Shared text-cleaning + phishing helpers (used by train.py and app.py)."""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                  ("stopwords", "corpora/stopwords")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english")) - {"won", "free", "now"}  # keep spam-signal words
ps = PorterStemmer()
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)


def clean_text(text: str) -> str:
    """lowercase -> strip urls/punctuation -> tokenize -> stopwords -> stem."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = text.translate(str.maketrans("", "", string.punctuation + "₹£€$"))
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    words = [w for w in tokens if w not in STOP_WORDS and not w.isdigit() or w == "urltoken"]
    return " ".join(ps.stem(w) for w in words)


# ---------------- Phishing detection (rule based) ----------------
URGENCY = ["urgent", "immediately", "act now", "within 24 hours", "final notice",
           "account suspended", "account will be closed", "verify now", "limited time"]
CREDENTIALS = ["password", "otp", "cvv", "pin", "login", "sign in", "verify your account",
               "bank account", "credit card", "ssn", "aadhaar", "pan card", "kyc"]
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "is.gd", "cutt.ly", "rb.gy"]
IP_URL = re.compile(r"https?://\d{1,3}(\.\d{1,3}){3}")


def phishing_check(text: str):
    """Return (score 0-100, list_of_reasons)."""
    t = text.lower()
    reasons, score = [], 0
    urls = URL_RE.findall(text)
    if urls:
        score += 15; reasons.append(f"Contains {len(urls)} link(s)")
    if IP_URL.search(t):
        score += 30; reasons.append("Link uses a raw IP address")
    if any(s in t for s in SHORTENERS):
        score += 20; reasons.append("Uses a URL shortener")
    if any(w in t for w in URGENCY):
        score += 20; reasons.append("Urgent / threatening language")
    if any(w in t for w in CREDENTIALS):
        score += 25; reasons.append("Asks for sensitive info (password/OTP/bank/KYC)")
    if re.search(r"https?://\S*@", t):
        score += 15; reasons.append("Link contains '@' (disguised destination)")
    return min(score, 100), reasons
