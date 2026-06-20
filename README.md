# MLOps — Lending Club

> Pipeline MLOps complet de prédiction de défaut de paiement, réalisé dans le cadre du projet fil rouge ESGI.

---

## Problématique

**LendingClub** est une plateforme américaine de prêt entre particuliers (*peer-to-peer lending*). Sur les données 2007–2018, environ **20 % des prêts** se terminent en défaut de paiement (*charged off*), représentant des pertes significatives pour les investisseurs.

**Objectif :** classifier automatiquement les dossiers de prêt à risque à partir de ~150 variables (score FICO, DTI, ancienneté professionnelle, historique de crédit…) — tâche de **classification binaire**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Docker Compose                          │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐    │
│  │ Airflow  │───▶│  Train   │───▶│  MLflow  │◀──│   API    │    │
│  │(scheduler│    │(LightGBM │    │(tracking │   │(FastAPI) │    │
│  │ + DAGs)  │    │ XGBoost  │    │ registry)│   └────┬─────┘    │
│  └──────────┘    │   RF     │    └──────────┘        │          │
│                  │  LogReg) │                        │          │
│                  └──────────┘              ┌─────────▼───────┐  │
│                                            │    Frontend     │  │
│                                            │  (Streamlit)    │  │
│                                            └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Modules `src/lending/`

| Module | Rôle |
|---|---|
| `data.py` | Chargement et nettoyage du CSV brut |
| `features.py` | Encodage catégoriel, normalisation, pipeline sklearn |
| `train.py` | Entraînement baseline (Logistic Regression) |
| `train_models.py` | Comparaison de 4 modèles en CV 5-fold |
| `train_optuna.py` | Optimisation Optuna des hyperparamètres |
| `evaluate.py` | Évaluation sur le split test (AUC-ROC, rapport de classification) |
| `tracking.py` | Logging MLflow + promotion du champion |
| `api.py` | API FastAPI — `/predict`, `/predictions`, `/health` |
| `config.py` | Constantes centralisées (colonnes, chemins, noms de modèles) |

### Pipeline de données

```
Raw CSV (Kaggle)
    │  data.py
    ▼
Nettoyage + filtrage par année (2007–2012 train / 2014 eval)
    │  features.py
    ▼
Pipeline sklearn : OrdinalEncoder + StandardScaler
    │  train_models.py
    ▼
CV 5-fold → AUC-ROC par modèle
    │  tracking.py
    ▼
MLflow registry → alias @champion → API
```

### Résultats modèles (AUC-ROC, CV 5-fold)

| Modèle | AUC-ROC |
|---|---|
| **LightGBM** | **0.689** ✅ |
| Logistic Regression | 0.684 |
| XGBoost | 0.671 |
| Random Forest | 0.666 |

---

## Stack technique

| Couche | Technologie |
|---|---|
| ML | scikit-learn, LightGBM, XGBoost, Optuna |
| Tracking | MLflow (SQLite backend + artifact serving) |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Orchestration | Apache Airflow (LocalExecutor + PostgreSQL) |
| Packaging | uv (toutes les images Docker) |
| Conteneurisation | Docker Compose (profils `data` / `train` / `airflow`) |
| Qualité | ruff, mypy, pytest |

---

## Démarrage rapide

### Prérequis

- Docker + Docker Compose
- Compte Kaggle (token API pour le téléchargement des données)

### Configuration

```bash
cp .env.example .env
# Renseigner KAGGLE_USER et KAGGLE_TKN dans .env
```

### Deploy complet

```bash
make deploy-local
```

Lance dans l'ordre : build des images → téléchargement des données → entraînement → stack complète.

> **Premier déploiement ou volumes corrompus :**
> ```bash
> make deploy-reset   # purge les volumes MLflow et relance depuis zéro
> ```

### Services

| Service | URL |
|---|---|
| Frontend (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

```bash
make deploy-down   # arrêter la stack
```

---

## Développement local

```bash
make install       # crée le venv et installe toutes les dépendances
make data          # télécharge et prépare le dataset
make train         # entraîne la baseline
make train-models  # compare les 4 modèles (logs MLflow)
make train-optuna  # optimisation Optuna (N_TRIALS=50)
make evaluate      # évalue le champion sur le split test
make api           # lance l'API en mode rechargement auto
make frontend      # lance le frontend Streamlit
```

### Qualité

```bash
make check         # lint (ruff) + types (mypy) + tests (pytest)
make lint          # ruff uniquement
make type          # mypy uniquement
make test          # pytest uniquement
```

---

## Structure du projet

```
.
├── src/lending/          # Code source Python
├── tests/                # Tests unitaires et d'intégration
├── frontend/             # Application Streamlit
│   ├── app.py
│   └── .streamlit/config.toml
├── dags/                 # DAGs Airflow
├── docker/               # Dockerfiles (api, train, frontend, data, airflow)
├── data/                 # Données (gitignored)
│   ├── raw/
│   └── processed/
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```

---

*Projet réalisé par **Théo ELOY** — ESGI*
