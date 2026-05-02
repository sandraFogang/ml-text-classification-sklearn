"""
preprocessing.py
Chargement du dataset 20 Newsgroups et vectorisation TF-IDF.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


CATEGORIES = ["rec.sport.baseball", "rec.sport.hockey"]


def load_data():
    """Charge les articles baseball + hockey depuis 20 Newsgroups."""
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        random_state=42,
    )
    return data.data, data.target


def get_processed_data(max_features=200):
    """
    Charge, vectorise et divise les données.

    Le vectorizer est fit sur le corpus complet avant le split,
    ce qui reproduit exactement la procédure du notebook original.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)
    """
    texts, labels = load_data()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        decode_error="replace",
        strip_accents="unicode",
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(texts)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tfidf, labels, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer
