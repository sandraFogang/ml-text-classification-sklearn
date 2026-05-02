"""
preprocessing.py
Chargement du dataset 20 Newsgroups, lemmatisation et vectorisation TF-IDF.
"""

import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download("wordnet", quiet=True)

CATEGORIES = ["rec.sport.baseball", "rec.sport.hockey"]


def load_data():
    """Charge les articles baseball + hockey depuis 20 Newsgroups."""
    data = fetch_20newsgroups(
        subset="all",
        categories=CATEGORIES,
    )
    return data.data, data.target


def lemmatize(texts):
    """Lemmatise chaque texte (verbes uniquement, pos='v')."""
    lemmatizer = WordNetLemmatizer()
    return pd.Series(texts).apply(
        lambda x: " ".join(lemmatizer.lemmatize(word, "v") for word in x.split())
    )


def get_processed_data(max_features=200):
    """
    Charge, lemmatise, vectorise et divise les données.

    Reproduit exactement la procédure du notebook original :
    - Lemmatisation avant vectorisation
    - TF-IDF fit sur le corpus complet
    - split avec shuffle=False

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)
    """
    texts, labels = load_data()

    X_lemmatized = lemmatize(texts)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        decode_error="replace",
        strip_accents="unicode",
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(X_lemmatized)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_tfidf, labels, test_size=0.30, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=False
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer
