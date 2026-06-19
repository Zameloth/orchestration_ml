import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")
API_DOCS_URL = os.environ.get("API_DOCS_URL", "http://localhost:8000/docs")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")

_TERM = [" 36 months", " 60 months"]
_GRADE = list("ABCDEFG")
_SUB_GRADE = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
_EMP_LENGTH = [
    "< 1 year",
    "1 year",
    "2 years",
    "3 years",
    "4 years",
    "5 years",
    "6 years",
    "7 years",
    "8 years",
    "9 years",
    "10+ years",
]
_HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "ANY", "NONE", "OTHER"]
_VERIFICATION = ["Not Verified", "Source Verified", "Verified"]
_PURPOSE = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "other",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "moving",
    "vacation",
    "house",
    "wedding",
    "renewable_energy",
    "educational",
]

_PRESET_DEFAULT = {
    "loan_amnt": 10000,
    "int_rate": 12.0,
    "installment": 300,
    "annual_inc": 60000,
    "dti": 15.0,
    "delinq_2yrs": 0,
    "fico_range": (690, 694),
    "inq_last_6mths": 0,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 15000,
    "revol_util": 50.0,
    "total_acc": 20,
    "term": " 36 months",
    "grade": "C",
    "sub_grade": "C3",
    "emp_length": "5 years",
    "home_ownership": "RENT",
    "verification_status": "Not Verified",
    "purpose": "debt_consolidation",
}

_PRESET_GOOD = {
    "loan_amnt": 8000,
    "int_rate": 7.0,
    "installment": 180,
    "annual_inc": 85000,
    "dti": 6.0,
    "delinq_2yrs": 0,
    "fico_range": (730, 734),
    "inq_last_6mths": 0,
    "open_acc": 8,
    "pub_rec": 0,
    "revol_bal": 4000,
    "revol_util": 15.0,
    "total_acc": 14,
    "term": " 36 months",
    "grade": "A",
    "sub_grade": "A2",
    "emp_length": "5 years",
    "home_ownership": "MORTGAGE",
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
}

_PRESET_RISKY = {
    "loan_amnt": 28000,
    "int_rate": 25.0,
    "installment": 850,
    "annual_inc": 32000,
    "dti": 32.0,
    "delinq_2yrs": 3,
    "fico_range": (590, 594),
    "inq_last_6mths": 6,
    "open_acc": 22,
    "pub_rec": 2,
    "revol_bal": 42000,
    "revol_util": 88.0,
    "total_acc": 32,
    "term": " 60 months",
    "grade": "F",
    "sub_grade": "F3",
    "emp_length": "< 1 year",
    "home_ownership": "RENT",
    "verification_status": "Not Verified",
    "purpose": "small_business",
}

for key, val in _PRESET_DEFAULT.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _risk_band(prob: float) -> tuple[str, str, str, str]:
    if prob < 0.30:
        return "Faible", "#d4edda", "#155724", "✅"
    if prob < 0.60:
        return "Modéré", "#fff3cd", "#856404", "⚠️"
    return "Élevé", "#f8d7da", "#721c24", "🔴"


with st.sidebar:
    st.markdown("### Services")
    st.link_button("API — Documentation", API_DOCS_URL, use_container_width=True)
    st.link_button("MLflow — Tracking", MLFLOW_URL, use_container_width=True)
    st.link_button("Airflow — Orchestration", AIRFLOW_URL, use_container_width=True)
    st.divider()
    st.caption("Réalisé par **Théo ELOY**")

st.title("Lending Club — Prédiction de défaut")

tab_home, tab_pred, tab_hist = st.tabs(["Accueil", "Prédiction", "Historique"])

with tab_home:
    st.header("Problématique métier")
    st.markdown(
        """
        **LendingClub** est une plateforme américaine de prêt entre particuliers (*peer-to-peer lending*).
        Les emprunteurs soumettent une demande de prêt, et des investisseurs particuliers financent ces prêts
        en échange d'intérêts. Le risque principal pour l'investisseur est le **défaut de paiement** :
        l'emprunteur ne rembourse pas son prêt.

        L'enjeu est considérable : sur les données 2007–2018, environ **20 % des prêts accordés**
        se terminent en défaut (*charged off*), représentant des pertes significatives pour les investisseurs.

        **Objectif de ce projet :** construire un modèle de classification binaire capable de prédire,
        à partir des caractéristiques d'un dossier de prêt, si l'emprunteur fera défaut —
        permettant ainsi de mieux sélectionner les prêts à financer.
        """
    )

    st.divider()
    st.header("Le dataset")
    st.markdown(
        """
        Les données proviennent du dataset public **Lending Club** disponible sur Kaggle.
        Il regroupe l'ensemble des prêts émis entre **2007 et 2018**.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Prêts enregistrés", "2 260 668")
    col2.metric("Variables disponibles", "~150")
    col3.metric("Taux de défaut", "~20 %")

    st.markdown(
        """
        Les variables clés utilisées pour la prédiction :

        | Variable | Description |
        |---|---|
        | `loan_amnt` | Montant du prêt demandé |
        | `int_rate` | Taux d'intérêt appliqué |
        | `fico_range_low/high` | Score de crédit FICO de l'emprunteur |
        | `dti` | Ratio dette / revenu annuel |
        | `annual_inc` | Revenu annuel déclaré |
        | `grade` / `sub_grade` | Note de risque attribuée par LendingClub |
        | `purpose` | Motif du prêt (consolidation de dettes, immobilier…) |
        | `delinq_2yrs` | Nombre d'incidents de paiement sur 2 ans |
        """
    )

    st.divider()
    st.header("Approche Machine Learning")
    st.markdown(
        """
        Le pipeline comprend les étapes suivantes :
        1. **Préparation des données** — nettoyage, encodage des variables catégorielles, normalisation
        2. **Entraînement** — comparaison de 4 modèles en validation croisée 5-fold
        3. **Sélection** — le meilleur modèle (AUC-ROC) est enregistré via **MLflow** et exposé par l'API
        4. **Serving** — API FastAPI + interface Streamlit (cette application)
        """
    )

    st.subheader("Comparaison des modèles (AUC-ROC, CV 5-fold)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LightGBM", "0.689", delta="meilleur", delta_color="normal")
    m2.metric("Logistic Reg.", "0.684")
    m3.metric("XGBoost", "0.671")
    m4.metric("Random Forest", "0.666")

    chart_data = pd.DataFrame(
        {"AUC-ROC": [0.689, 0.684, 0.671, 0.666]},
        index=["LightGBM", "Logistic Reg.", "XGBoost", "Random Forest"],
    )
    st.bar_chart(chart_data)

with tab_pred:
    st.caption("Charger un profil type :")
    p1, p2, p3 = st.columns(3)
    if p1.button("Profil par défaut", use_container_width=True):
        for k, v in _PRESET_DEFAULT.items():
            st.session_state[k] = v
    if p2.button("Bon emprunteur", use_container_width=True):
        for k, v in _PRESET_GOOD.items():
            st.session_state[k] = v
    if p3.button("Emprunteur à risque", use_container_width=True):
        for k, v in _PRESET_RISKY.items():
            st.session_state[k] = v

    with st.form("prediction_form"):
        st.subheader("Caractéristiques du prêt")
        c1, c2 = st.columns(2)
        with c1:
            loan_amnt = st.slider("Montant du prêt ($)", 500, 40000, step=500, key="loan_amnt")
            int_rate = st.slider("Taux d'intérêt (%)", 5.0, 30.0, step=0.5, key="int_rate")
            installment = st.slider("Mensualité ($)", 10, 1500, step=10, key="installment")
        with c2:
            term = st.selectbox("Durée du prêt", _TERM, key="term")
            purpose = st.selectbox("Motif du prêt", _PURPOSE, key="purpose")
            grade = st.selectbox("Note de risque", _GRADE, key="grade")
            sub_grade = st.selectbox("Sous-note", _SUB_GRADE, key="sub_grade")

        st.subheader("Profil financier")
        c3, c4 = st.columns(2)
        with c3:
            annual_inc = st.slider("Revenu annuel ($)", 0, 500_000, step=5000, key="annual_inc")
            dti = st.slider("Ratio dette/revenu (%)", 0.0, 60.0, step=0.5, key="dti")
            fico_range = st.slider("Score FICO", 580, 854, step=4, key="fico_range")
        with c4:
            revol_bal = st.slider("Solde revolving ($)", 0, 150_000, step=1000, key="revol_bal")
            revol_util = st.slider(
                "Utilisation revolving (%)", 0.0, 100.0, step=1.0, key="revol_util"
            )
            emp_length = st.selectbox("Ancienneté professionnelle", _EMP_LENGTH, key="emp_length")
            home_ownership = st.selectbox(
                "Statut résidentiel", _HOME_OWNERSHIP, key="home_ownership"
            )
            verification_status = st.selectbox(
                "Vérification du revenu", _VERIFICATION, key="verification_status"
            )

        st.subheader("Historique de crédit")
        c5, c6 = st.columns(2)
        with c5:
            delinq_2yrs = st.slider(
                "Incidents de paiement (2 ans)", 0, 20, step=1, key="delinq_2yrs"
            )
            inq_last_6mths = st.slider(
                "Demandes de crédit (6 mois)", 0, 15, step=1, key="inq_last_6mths"
            )
            pub_rec = st.slider("Mentions publiques (faillites…)", 0, 10, step=1, key="pub_rec")
        with c6:
            open_acc = st.slider("Comptes de crédit ouverts", 0, 60, step=1, key="open_acc")
            total_acc = st.slider("Total lignes de crédit", 1, 100, step=1, key="total_acc")

        submitted = st.form_submit_button("Prédire", use_container_width=True)

    if submitted:
        fico_low, fico_high = (
            fico_range if isinstance(fico_range, tuple) else (fico_range, fico_range + 4)
        )
        payload = {
            "loan_amnt": float(loan_amnt),
            "int_rate": float(int_rate),
            "installment": float(installment),
            "annual_inc": float(annual_inc),
            "dti": float(dti),
            "delinq_2yrs": float(delinq_2yrs),
            "fico_range_low": float(fico_low),
            "fico_range_high": float(fico_high),
            "inq_last_6mths": float(inq_last_6mths),
            "open_acc": float(open_acc),
            "pub_rec": float(pub_rec),
            "revol_bal": float(revol_bal),
            "revol_util": float(revol_util),
            "total_acc": float(total_acc),
            "term": term,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "verification_status": verification_status,
            "purpose": purpose,
        }
        try:
            resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
            resp.raise_for_status()
            result = resp.json()

            prob = result["default_probability"]
            risk_label, bg_color, text_color, icon = _risk_band(prob)

            st.divider()
            st.markdown(
                f"""
                <div style="background-color:{bg_color}; color:{text_color}; padding:1.5rem;
                            border-radius:0.5rem; margin-bottom:1rem;">
                    <h3 style="margin:0 0 0.5rem 0;">{icon} Niveau de risque : {risk_label}</h3>
                    <p style="margin:0; font-size:1.1rem;">
                        Probabilité de défaut : <strong>{prob:.1%}</strong> —
                        Classe prédite : <strong>{result["prediction"]}</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(prob)

        except httpx.HTTPError as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

with tab_hist:
    if st.button("Rafraîchir"):
        st.rerun()

    try:
        resp = httpx.get(f"{API_URL}/predictions", timeout=10.0)
        resp.raise_for_status()
        records = resp.json()

        if not records:
            st.info("Aucune prédiction enregistrée pour l'instant.")
        else:
            df = pd.DataFrame(records)

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                st.subheader("Distribution des probabilités")
                st.bar_chart(df["default_probability"].value_counts(bins=10).sort_index())

            with col_chart2:
                st.subheader("Répartition des prédictions")
                pie_counts = df["prediction"].value_counts()
                st.bar_chart(pie_counts)

            st.divider()
            df_display = df.copy()
            df_display["default_probability"] = df_display["default_probability"].map(
                "{:.1%}".format
            )
            df_display = df_display.rename(
                columns={
                    "id": "ID",
                    "created_at": "Date",
                    "loan_amnt": "Montant",
                    "grade": "Note",
                    "int_rate": "Taux (%)",
                    "purpose": "Motif",
                    "default_probability": "Prob. défaut",
                    "prediction": "Prédiction",
                }
            )
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.caption(f"{len(df)} prédiction(s) au total")

    except httpx.HTTPError as e:
        st.error(f"Erreur lors de la récupération de l'historique : {e}")
