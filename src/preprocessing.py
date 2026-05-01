"""
preprocessing.py
Chargement du dataset 20 Newsgroups et vectorisation TF-IDF.
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


CATEGORIES = ["rec.sport.baseball", "rec.sport.hockey"]
RANDOM_STATE = 42


def load_data():
    """Charge les articles baseball + hockey depuis 20 Newsgroups."""
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
        random_state=RANDOM_STATE,
    )
    return data.data, data.target


def split_data(X, y):
    """
    Divise en train (70%) / val (15%) / test (15%).
    Deux appels à train_test_split pour reproduire la procédure originale.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_tfidf(X_train, X_val, X_test, max_features=200):
    """
    Ajuste un TfidfVectorizer sur le train et transforme les trois ensembles.
    Paramètres identiques à ceux utilisés dans l'analyse.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        decode_error="replace",
        strip_accents="unicode",
        stop_words="english",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def get_processed_data(max_features=200):
    """
    Point d'entrée unique : charge, divise et vectorise les données.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)
    """
    texts, labels = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = build_tfidf(
        X_train, X_val, X_test, max_features=max_features
    )
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, y_train, y_val, y_test, vectorizer
