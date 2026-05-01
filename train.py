"""
train.py
Script principal : charge les données, entraîne les 3 modèles,
évalue sur le test et génère les figures + metrics.json.

Usage
-----
    python train.py
"""

from src.preprocessing import get_processed_data
from src.models import train_naive_bayes, train_svm, train_mlp
from src.evaluate import (
    evaluate_on_test,
    plot_confusion_matrix,
    plot_top_tfidf_features,
    plot_mlp_sensitivity,
    plot_model_comparison,
    save_metrics,
)


def main():
    print("Loading and vectorizing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = get_processed_data()

    print("\nTraining Naive Bayes...")
    nb_model, nb_summary = train_naive_bayes(X_train, y_train, X_val, y_val)

    print("\nTraining SVM (grid search: 9 combinations)...")
    svm_model, svm_summary = train_svm(X_train, y_train, X_val, y_val)

    print("\nTraining MLP (grid search: 81 combinations — ~5-10 min)...")
    mlp_model, mlp_summary = train_mlp(X_train, y_train, X_val, y_val)

    summaries = [nb_summary, svm_summary, mlp_summary]
    print("\n--- Validation results ---")
    for s in summaries:
        print(s)

    print("\n--- Test evaluation ---")
    nb_test_acc, _  = evaluate_on_test(nb_model,  X_test, y_test, "Naive Bayes")
    svm_test_acc, _ = evaluate_on_test(svm_model, X_test, y_test, "SVM")
    mlp_test_acc, _ = evaluate_on_test(mlp_model, X_test, y_test, "MLP")

    test_accuracies = {
        "Naive Bayes": nb_test_acc,
        "SVM": svm_test_acc,
        "MLP": mlp_test_acc,
    }

    print("\nGenerating figures...")
    plot_confusion_matrix(svm_model, X_test, y_test, model_name="SVM")
    plot_top_tfidf_features(vectorizer)
    plot_mlp_sensitivity(X_train, y_train, X_val, y_val)
    plot_model_comparison(summaries, test_accuracies)
    save_metrics(summaries, test_accuracies)

    print("\nDone. Outputs saved in outputs/")


if __name__ == "__main__":
    main()
