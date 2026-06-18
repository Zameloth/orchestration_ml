# ==============================================================================
# Projet de classification - Makefile (squelette)
# ==============================================================================
# Seuls les targets d'INSTALLATION sont fournis. Les autres sont a completer
# au fil des TP (un `# TODO (Sx)` indique la commande attendue).
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
PYTHON       := uv run python
RUN          := uv run
VENV_DIR     := .venv
PYTHONPATH   ?= .
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5000
AIRFLOW_PORT ?= 8080
REPO         ?= zameloth/orchestration_ml
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna evaluate mlflow api frontend \
        docker-build docker-run docker-up docker-down \
        deploy-local-build deploy-local deploy-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML  [A COMPLETER]
# ==============================================================================

RAW_CSV := data/raw/accepted_2007_to_2018q4.csv

data: $(RAW_CSV) ## Telecharge et extrait le dataset Lending Club (Kaggle)

$(RAW_CSV):
	@echo "$(YELLOW)>> Telechargement du dataset Lending Club...$(RESET)"
	@command -v kaggle >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] kaggle CLI absent. Installe-le : pip install kaggle$(RESET)"; \
		echo "  Puis configure ~/.kaggle/kaggle.json (https://www.kaggle.com/docs/api)"; \
		exit 1; \
	}
	mkdir -p data/raw
	kaggle datasets download -d wordsforthewise/lending-club -p data/raw/ --unzip
	@echo "$(GREEN)[OK] Dataset disponible : $(RAW_CSV)$(RESET)"

train: ## Entraine la baseline -> data/models/baseline.joblib
	$(PYTHON) -m lending.train

train-models: ## Compare RF / XGBoost / LightGBM avec CV (CV=.. SCORING=..)
	$(PYTHON) -m lending.train_models --cv $(CV) --scoring $(SCORING)

train-optuna: ## Optimise RF / XGBoost / LightGBM avec Optuna (N_TRIALS=.. CV=..)
	$(PYTHON) -m lending.train_optuna --n-trials $(N_TRIALS) --cv $(CV)

evaluate: ## Evalue le meilleur modele sur les donnees de test
	$(PYTHON) -m lending.evaluate

mlflow: ## Lance l'UI MLflow locale (sqlite)
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port $(MLFLOW_PORT)

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	$(RUN) uvicorn lending.api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT, API_URL)
	$(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker  [A COMPLETER]
# ==============================================================================

docker-build: ## Construit les images (train + api)
	docker compose -f docker-compose.yml build

docker-run: ## Lance l'entrainement one-shot (profil train)
	docker compose -f docker-compose.yml --profile train run --rm train

docker-up: ## Demarre la stack (mlflow + api + frontend)
	docker compose -f docker-compose.yml up -d mlflow api frontend

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	docker compose -f docker-compose.yml down


# ==============================================================================
# Deploy local  (équivalent CD)
# ==============================================================================

deploy-local-build: ## Construit toutes les images Docker en local
	@echo "$(YELLOW)>> Build des images Docker...$(RESET)"
	docker build -f docker/Dockerfile.api      -t ghcr.io/$(REPO)/mlproject-api:latest      .
	docker build -f docker/Dockerfile.frontend -t ghcr.io/$(REPO)/mlproject-frontend:latest .
	docker compose build
	@echo "$(GREEN)[OK] Images construites$(RESET)"

deploy-local: deploy-local-build ## Deploy complet en local : build → data → train → stack (airflow + api + frontend)
	@echo "$(YELLOW)>> Pipeline data (KAGGLE_USER + KAGGLE_TKN depuis .env requis)...$(RESET)"
	docker compose --profile data run --rm data
	@echo "$(YELLOW)>> Entrainement du modele...$(RESET)"
	docker compose --profile train run --rm train
	@echo "$(YELLOW)>> Demarrage de la stack (mlflow + api + frontend + airflow)...$(RESET)"
	docker compose --profile airflow up -d --remove-orphans
	@echo "$(GREEN)[OK] Deploy local termine$(RESET)"
	@printf "  MLflow   : http://localhost:$(MLFLOW_PORT)\n"
	@printf "  API      : http://localhost:$(API_PORT)\n"
	@printf "  Frontend : http://localhost:$(FRONTEND_PORT)\n"
	@printf "  Airflow  : http://localhost:$(AIRFLOW_PORT)\n"

deploy-down: ## Arrete toute la stack de deploy local (tous les profils)
	docker compose --profile airflow --profile data --profile train down


# ==============================================================================
# Qualite  [A COMPLETER]
# ==============================================================================

lint: ## Verifie le style (ruff)
	$(RUN) ruff check src/

format: ## Formate le code (ruff)
	$(RUN) ruff format src/

type: ## Verifie les types (mypy)
	$(RUN) mypy src/

test: ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)
