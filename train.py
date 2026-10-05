import json
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


# =========================
# NLP PREPROCESSING
# =========================

def preprocess_text(text):

    text = text.lower()

    # Common contractions
    replacements = {
        "what's": "what is",
        "whats": "what is",
        "who's": "who is",
        "who're": "who are",
        "how's": "how is",
        "how're": "how are",
        "can't": "cannot",
        "don't": "do not",
        "doesn't": "does not",
        "isn't": "is not",
        "aren't": "are not",
        "i'm": "i am",
        "im": "i am",
        "you're": "you are",
        "youre": "you are",
        "it's": "it is",
        "its": "it is"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Common informal words
    informal_words = {
        "abt": "about",
        "pls": "please",
        "plz": "please",
        "u": "you",
        "ur": "your",
        "r": "are",
        "wat": "what",
        "wht": "what",
        "hw": "how"
    }

    words = text.split()

    words = [
        informal_words.get(word, word)
        for word in words
    ]

    text = " ".join(words)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================
# LOAD DATA
# =========================

with open(
    "intents.json",
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


patterns = []
tags = []


# =========================
# PREPROCESS TRAINING DATA
# =========================

for intent in data["intents"]:

    if intent["tag"] == "fallback":
        continue

    for pattern in intent["patterns"]:

        cleaned_pattern = preprocess_text(
            pattern
        )

        patterns.append(
            cleaned_pattern
        )

        tags.append(
            intent["tag"]
        )


# =========================
# WORD TF-IDF
# =========================

word_vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=5000
)


# =========================
# CHARACTER TF-IDF
# =========================

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    lowercase=True,
    ngram_range=(3, 5),
    sublinear_tf=True,
    max_features=5000
)


# =========================
# COMBINE FEATURES
# =========================

vectorizer = FeatureUnion([
    ("word", word_vectorizer),
    ("char", char_vectorizer)
])


X = vectorizer.fit_transform(
    patterns
)


# =========================
# MACHINE LEARNING MODEL
# =========================

model = LogisticRegression(
    max_iter=2000,
    C=5.0,
    class_weight="balanced"
)


model.fit(
    X,
    tags
)


# =========================
# SAVE MODEL
# =========================

with open(
    "chatbot_model.pkl",
    "wb"
) as file:

    pickle.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        file
    )


# =========================
# INFORMATION
# =========================

print("======================================")
print("       SMARTCHAT AI MODEL TRAINED")
print("======================================")

print(
    f"Training examples : {len(patterns)}"
)

print(
    f"Number of intents : {len(set(tags))}"
)

print(
    f"Features          : {X.shape[1]}"
)

print(
    "NLP preprocessing  : Enabled"
)

print(
    "Word TF-IDF        : Enabled"
)

print(
    "Character TF-IDF   : Enabled"
)

print(
    "ML Algorithm       : Logistic Regression"
)

print(
    "Model saved        : chatbot_model.pkl"
)

print("======================================")