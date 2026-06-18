# MLOps — Lending Club (ESGI)

Projet fil rouge d'orchestration Machine Learning basé sur le dataset **Lending Club**.

## Dataset

**Source :** [Lending Club — Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Le dataset Lending Club regroupe l'ensemble des prêts émis entre 2007 et 2018 par la plateforme de prêt entre particuliers LendingClub. Il contient plus de **2 millions de lignes** et une centaine de variables décrivant chaque demande de prêt (montant, taux d'intérêt, durée, score de crédit, historique de paiement, etc.).

**Objectif :** prédire si un emprunteur va rembourser son prêt ou faire défaut (`loan_status`) — tâche de **classification binaire**.

## Installation

```bash
make install
```

## Commandes principales

```bash
make help          # Liste toutes les commandes disponibles
make data          # Prépare le jeu de données
make train         # Entraîne le modèle baseline
make train-models  # Compare RF / XGBoost / LightGBM
make mlflow        # Démarre le serveur MLflow
make api           # Lance l'API FastAPI
make frontend      # Lance le frontend Streamlit
```

## Deploy local (équivalent CD)

Reproduit en local le pipeline complet de la CI/CD : build des images Docker, téléchargement des données, entraînement, puis démarrage de la stack complète (MLflow, API, Streamlit, Airflow).

**Prérequis :** copier `.env.example` en `.env` et renseigner les credentials Kaggle.

```bash
cp .env.example .env
# éditer .env : renseigner KAGGLE_USER et KAGGLE_TKN

make deploy-local
```

Une fois la stack démarrée :

| Service  | URL                        |
|----------|----------------------------|
| MLflow   | http://localhost:5000       |
| API      | http://localhost:8000       |
| Frontend | http://localhost:8501       |
| Airflow  | http://localhost:8080       |

Pour arrêter la stack :

```bash
make deploy-down
```

> **Note :** `make deploy-local-build` construit uniquement les images sans lancer le pipeline, utile pour vérifier que les Dockerfiles compilent correctement.
