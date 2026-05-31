# airquality
Prédiction des concentrations de polluants atmosphériques (CO, NO₂, NOx, C6H6) à partir du dataset AirQualityUCI — comparaison de régressions linéaires, Ridge, Lasso, Kernel Ridge (RBF) et réseau de neurones dense.

[![Voir le notebook](https://img.shields.io/badge/Jupyter-Voir%20le%20notebook-orange?logo=jupyter)](https://nbviewer.org/github/el7abib/airquality/blob/main/airquality.ipynb)


# Prédiction de la Qualité de l'Air — AirQualityUCI

Analyse et modélisation de données de qualité de l'air à partir du dataset **AirQualityUCI**, avec comparaison de plusieurs approches de régression.

---

## Objectif

Prédire les concentrations de quatre polluants atmosphériques à partir de données de capteurs et de mesures environnementales :

- **CO(GT)** — Monoxyde de carbone (référence)
- **C6H6(GT)** — Benzène
- **NOx(GT)** — Oxydes d'azote
- **NO2(GT)** — Dioxyde d'azote

---

## Dataset

**Source :** [AirQualityUCI.csv](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

Le dataset contient des mesures horaires effectuées par une station de surveillance de la qualité de l'air en Italie. Il inclut des relevés de capteurs chimiques multi-gaz ainsi que des mesures de référence et des variables environnementales (température, humidité relative et absolue).

Les valeurs manquantes sont codées **-200** dans le dataset original.

---

## Structure du notebook

### 1. Nettoyage des données
- Chargement et nettoyage via `skrub.Cleaner`
- Suppression des colonnes vides et des lignes contenant des valeurs nulles
- Remplacement des séparateurs décimaux (`,` → `.`) et conversion en `float`
- Remplacement des `-200` par `NaN` pour le traitement des valeurs manquantes

### 2. Analyse exploratoire (EDA)
- **ACP (PCA)** : réduction de dimension et cercle de corrélation pour visualiser les groupes de variables
- **Heatmap des valeurs manquantes** : localisation des `-200` dans le dataset
- **Distributions et évolution horaire** : histogrammes + KDE et courbes d'évolution par heure de la journée pour les 4 polluants cibles
- **TableReport (skrub)** : analyse détaillée des associations entre variables

Observation clé : les concentrations de polluants présentent **deux pics journaliers** (matin et fin de journée), corrélés avec le trafic routier.

### 3. Modélisation

Le dataset est découpé en trois ensembles :
| Ensemble | Proportion |
|----------|-----------|
| Train    | 70%       |
| Validation | 13%    |
| Test     | 17%       |

#### Modèles comparés

| Modèle | Description |
|--------|-------------|
| **Régression Linéaire** | Baseline sans pénalisation |
| **Ridge** | Pénalisation L2, sélection d'alpha par validation |
| **Lasso** | Pénalisation L1, alpha optimal très faible (≈ régression linéaire) |
| **Kernel Ridge — Linéaire** | Équivalent Ridge dans un RKHS à noyau linéaire |
| **Kernel Ridge — RBF** | Noyau gaussien, capture les relations non linéaires |
| **Réseau de neurones dense** | MLP avec optimisation de l'architecture (couches, neurones, LR, epochs) |

#### Résultats clés

Les modèles linéaires (LR, Ridge, Lasso) obtiennent des performances similaires car l'alpha optimal de Ridge/Lasso est très faible. Tous trois présentent une **courbure systématique** dans les graphes True vs Predicted, révélant que la relation entre les variables est non linéaire.

Le **Kernel Ridge RBF** et le **réseau de neurones** surpassent nettement les modèles linéaires, le Kernel RBF offrant le meilleur compromis performances/complexité.

### 4. Amélioration — TimeSeriesSplit

Les données étant des mesures horaires consécutives, un découpage aléatoire introduit un **biais temporel** (le modèle "voit" le futur lors de l'entraînement). Un `TimeSeriesSplit` est implémenté pour corriger ce problème :
- Entraînement sur les données passées uniquement
- Validation et test sur des données strictement ultérieures
- Résultat : performances légèrement inférieures mais **évaluation plus réaliste**

---

## Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn skrub tensorflow
```

---

## Utilisation

Ouvrir et exécuter le notebook `appstat_final__3_.ipynb` cellule par cellule. S'assurer que le fichier `AirQualityUCI.csv` est présent dans le même répertoire.

```bash
jupyter notebook appstat_final__3_.ipynb
```

---

## Métriques d'évaluation

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient de détermination

---

## Dépendances principales

| Bibliothèque | Usage |
|---|---|
| `scikit-learn` | Modèles, pipelines, métriques, validation croisée |
| `pandas` / `numpy` | Manipulation des données |
| `matplotlib` / `seaborn` | Visualisations |
| `skrub` | Nettoyage automatique et rapport des données |
| `tensorflow` / `keras` | Réseau de neurones dense |
