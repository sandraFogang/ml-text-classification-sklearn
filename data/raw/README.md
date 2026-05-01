# Données

## Source

**20 Newsgroups** — dataset public de référence en traitement automatique du langage.

- Hébergé et distribué via scikit-learn : `sklearn.datasets.fetch_20newsgroups`
- Catégories utilisées : `rec.sport.baseball` et `rec.sport.hockey`
- Sous-ensemble : `subset="all"` (train + test originaux combinés, puis re-divisés)

## Chargement

Les données se téléchargent automatiquement au premier lancement de `train.py` :

```python
from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups(
    subset="all",
    categories=["rec.sport.baseball", "rec.sport.hockey"],
    random_state=42,
)
```

Aucun fichier de données n'est inclus dans ce repo.

## Partition

| Ensemble | Proportion |
|----------|-----------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Split effectué avec `random_state=42` pour reproductibilité.

## Représentation

Vectorisation TF-IDF avec les paramètres suivants :

| Paramètre | Valeur |
|-----------|--------|
| `max_features` | 200 |
| `decode_error` | `"replace"` |
| `strip_accents` | `"unicode"` |
| `stop_words` | `"english"` |
