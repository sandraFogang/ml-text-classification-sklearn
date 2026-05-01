"""
evaluate.py
Évaluation des modèles sur le test, visualisations et export des métriques.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


FIGURES_DIR = Path("outputs/figures")
METRICS_PATH = Path("outputs/metrics.json")
CATEGORIES = ["baseball", "hockey"]


def evaluate_on_test(model, X_test, y_test, model_name):
    """Calcule accuracy et rapport de classification sur le test."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=True)
    print(f"\n{model_name} — test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    return round(acc, 6), report


def plot_confusion_matrix(model, X_test, y_test, model_name="SVM", save=True):
    """
    Matrice de confusion normalisée du modèle retenu.
    Montre la répartition des erreurs au-delà de l'accuracy globale.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CATEGORIES,
        yticklabels=CATEGORIES,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name} — confusion matrix (normalized)")
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / f"confusion_matrix_{model_name.lower()}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()


def plot_top_nb_features(nb_model, vectorizer, n=10, save=True):
    """
    Top n termes les plus impactants selon Naïve Bayes, par classe.
    Utilise les log-probabilités conditionnelles du modèle NB :
    un score élevé indique que le terme est fortement associé à cette classe.
    Deux sous-graphes côte à côte : baseball (gauche) et hockey (droite).
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    # nb_model.feature_log_prob_ : shape (n_classes, n_features)
    log_probs = nb_model.feature_log_prob_

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    colors = ["#4C72B0", "#DD8452"]

    for i, (label, color) in enumerate(zip(CATEGORIES, colors)):
        top_idx = np.argsort(log_probs[i])[-n:]
        top_terms = feature_names[top_idx]
        top_scores = log_probs[i][top_idx]

        axes[i].barh(top_terms, top_scores, color=color)
        axes[i].set_title(f"Top {n} — {label}", fontsize=10)
        axes[i].set_xlabel("Log-probability", fontsize=8)
        axes[i].tick_params(labelsize=8)

    fig.suptitle("Naïve Bayes — most impactful terms per class", fontsize=11)
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "nb_top_features.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'nb_top_features.png'}")
    plt.close()


def plot_mlp_sensitivity(X_train, y_train, X_val, y_val, save=True):
    """
    Heatmap val accuracy : learning rate × régularisation L2.
    Couche 1 fixée à 4 neurones, couche 2 absente (config optimale).
    """
    from sklearn.neural_network import MLPClassifier

    learning_rates = [0.001, 0.01, 0.1]
    l2_strengths = [0.001, 0.01, 0.1]
    val_accs = np.zeros((len(learning_rates), len(l2_strengths)))

    for i, lr in enumerate(learning_rates):
        for j, l2 in enumerate(l2_strengths):
            model = MLPClassifier(
                hidden_layer_sizes=(4,),
                activation="logistic",
                alpha=l2,
                learning_rate_init=lr,
                early_stopping=True,
                random_state=12345,
                max_iter=500,
            )
            model.fit(X_train, y_train)
            val_accs[i, j] = accuracy_score(y_val, model.predict(X_val))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(
        val_accs,
        annot=True,
        fmt=".3f",
        xticklabels=l2_strengths,
        yticklabels=learning_rates,
        cmap="RdYlGn",
        ax=ax,
        vmin=0.5,
        vmax=1.0,
    )
    ax.set_xlabel("L2 regularization")
    ax.set_ylabel("Learning rate")
    ax.set_title("MLP — validation accuracy (lr × L2)")
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "mlp_heatmap_lr_l2.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'mlp_heatmap_lr_l2.png'}")
    plt.close()


def plot_model_comparison(summaries, test_accuracies, save=True):
    """Graphique en barres comparant val et test accuracy des 3 modèles."""
    names = [s["model"] for s in summaries]
    val_accs = [s["val_accuracy"] for s in summaries]
    test_accs = [test_accuracies.get(s["model"], 0) for s in summaries]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_val = ax.bar(x - width / 2, val_accs, width, label="Validation", color="#4C72B0")
    bars_test = ax.bar(x + width / 2, test_accs, width, label="Test", color="#DD8452")

    ax.set_ylabel("Accuracy")
    ax.set_title("Comparison of models — validation vs test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0.85, 1.0)
    ax.legend()
    ax.bar_label(bars_val, fmt="%.3f", padding=2, fontsize=9)
    ax.bar_label(bars_test, fmt="%.3f", padding=2, fontsize=9)
    fig.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
        print(f"Saved: {FIGURES_DIR / 'model_comparison.png'}")
    plt.close()


def save_metrics(summaries, test_accuracies):
    """Exporte les métriques val + test dans outputs/metrics.json."""
    results = []
    for s in summaries:
        entry = {**s, "test_accuracy": test_accuracies.get(s["model"], None)}
        results.append(entry)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {METRICS_PATH}")
