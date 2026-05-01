"""
models.py
Entraînement et sélection des trois modèles : Naïve Bayes, SVM, MLP.
Chaque fonction retourne le meilleur modèle et un résumé de ses paramètres.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------------
# Naïve Bayes
# ---------------------------------------------------------------------------

def train_naive_bayes(X_train, y_train, X_val, y_val):
    """
    Entraîne un Naïve Bayes multinomial.
    Pas de sélection d'hyperparamètres — sert de baseline.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, model.predict(X_val))
    summary = {"model": "Naive Bayes", "val_accuracy": round(val_acc, 6)}
    return model, summary


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

def train_svm(X_train, y_train, X_val, y_val):
    """
    Recherche sur grille : C ∈ {0.1, 1, 10} × kernel ∈ {linear, rbf, poly}.
    Sélection par accuracy de validation.
    """
    C_values = [0.1, 1, 10]
    kernels = ["linear", "rbf", "poly"]

    best_model = None
    best_val_acc = 0
    best_params = {}

    for c in C_values:
        for k in kernels:
            model = SVC(C=c, kernel=k, random_state=42)
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_params = {"C": c, "kernel": k}

    summary = {
        "model": "SVM",
        **best_params,
        "val_accuracy": round(best_val_acc, 6),
    }
    return best_model, summary


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def train_mlp(X_train, y_train, X_val, y_val):
    """
    Recherche exhaustive sur 81 combinaisons d'hyperparamètres :
      - taille couche 1 : {4, 8, 16}
      - taille couche 2 : {0, 4, 8}  (0 = une seule couche)
      - learning rate   : {0.001, 0.01, 0.1}
      - régularisation L2 : {0.001, 0.01, 0.1}
    """
    layer1_sizes = [4, 8, 16]
    layer2_sizes = [0, 4, 8]
    learning_rates = [0.001, 0.01, 0.1]
    l2_strengths = [0.001, 0.01, 0.1]

    best_model = None
    best_val_acc = 0
    best_params = {}

    for h1 in layer1_sizes:
        for h2 in layer2_sizes:
            for lr in learning_rates:
                for l2 in l2_strengths:
                    layers = (h1,) if h2 == 0 else (h1, h2)
                    model = MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation="logistic",
                        alpha=l2,
                        learning_rate_init=lr,
                        early_stopping=True,
                        random_state=12345,
                        max_iter=500,
                    )
                    model.fit(X_train, y_train)
                    val_acc = accuracy_score(y_val, model.predict(X_val))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model
                        best_params = {
                            "layer1": h1,
                            "layer2": h2,
                            "learning_rate": lr,
                            "l2": l2,
                        }

    summary = {
        "model": "MLP",
        **best_params,
        "val_accuracy": round(best_val_acc, 6),
    }
    return best_model, summary
