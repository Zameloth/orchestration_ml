import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

_TERM = [" 36 months", " 60 months"]
_GRADE = list("ABCDEFG")
_SUB_GRADE = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
_EMP_LENGTH = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years",
]
_HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "ANY", "NONE", "OTHER"]
_VERIFICATION = ["Not Verified", "Source Verified", "Verified"]
_PURPOSE = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "small_business", "car", "medical", "moving",
    "vacation", "house", "wedding", "renewable_energy", "educational",
]

st.title("Lending Club — Prédiction de défaut")

tab_pred, tab_hist = st.tabs(["Prédiction", "Historique"])

with tab_pred:
    with st.form("prediction_form"):
        st.subheader("Informations du prêt")
        c1, c2 = st.columns(2)
        with c1:
            loan_amnt = st.number_input("Montant (loan_amnt)", 500.0, 40000.0, 10000.0, 500.0)
            int_rate = st.number_input("Taux d'intérêt % (int_rate)", 5.0, 30.0, 12.0, 0.1)
            installment = st.number_input("Mensualité (installment)", 10.0, 1500.0, 300.0, 10.0)
            annual_inc = st.number_input("Revenu annuel (annual_inc)", 0.0, 1_000_000.0, 60_000.0, 1000.0)
            dti = st.number_input("Ratio dette/revenu (dti)", 0.0, 100.0, 15.0, 0.1)
            delinq_2yrs = st.number_input("Incidents 2 ans (delinq_2yrs)", 0.0, 30.0, 0.0, 1.0)
            fico_range_low = st.number_input("FICO bas (fico_range_low)", 580.0, 850.0, 690.0, 5.0)
        with c2:
            fico_range_high = st.number_input("FICO haut (fico_range_high)", 584.0, 854.0, 694.0, 5.0)
            inq_last_6mths = st.number_input("Demandes crédit 6 mois (inq_last_6mths)", 0.0, 33.0, 0.0, 1.0)
            open_acc = st.number_input("Comptes ouverts (open_acc)", 0.0, 90.0, 10.0, 1.0)
            pub_rec = st.number_input("Mentions publiques (pub_rec)", 0.0, 86.0, 0.0, 1.0)
            revol_bal = st.number_input("Solde revolving (revol_bal)", 0.0, 300_000.0, 15_000.0, 500.0)
            revol_util = st.number_input("Utilisation revolving % (revol_util)", 0.0, 100.0, 50.0, 1.0)
            total_acc = st.number_input("Total comptes (total_acc)", 1.0, 176.0, 20.0, 1.0)

        st.subheader("Profil emprunteur")
        c3, c4 = st.columns(2)
        with c3:
            term = st.selectbox("Durée (term)", _TERM)
            grade = st.selectbox("Note (grade)", _GRADE)
            sub_grade = st.selectbox("Sous-note (sub_grade)", _SUB_GRADE)
            emp_length = st.selectbox("Ancienneté emploi (emp_length)", _EMP_LENGTH)
        with c4:
            home_ownership = st.selectbox("Logement (home_ownership)", _HOME_OWNERSHIP)
            verification_status = st.selectbox("Vérification revenu (verification_status)", _VERIFICATION)
            purpose = st.selectbox("Motif du prêt (purpose)", _PURPOSE)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        payload = {
            "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
            "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq_2yrs,
            "fico_range_low": fico_range_low, "fico_range_high": fico_range_high,
            "inq_last_6mths": inq_last_6mths, "open_acc": open_acc, "pub_rec": pub_rec,
            "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
            "term": term, "grade": grade, "sub_grade": sub_grade,
            "emp_length": emp_length, "home_ownership": home_ownership,
            "verification_status": verification_status, "purpose": purpose,
        }
        try:
            resp = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
            resp.raise_for_status()
            result = resp.json()

            m1, m2 = st.columns(2)
            m1.metric("Classe prédite", result["prediction"])
            m2.metric("Probabilité de défaut", f"{result['default_probability']:.1%}")
            st.progress(result["default_probability"])

            if result["prediction"] == "charged_off":
                st.error("Risque élevé de défaut de paiement.")
            else:
                st.success("Faible risque de défaut de paiement.")

        except httpx.HTTPError as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")

with tab_hist:
    st.info("Historique non disponible (endpoint GET /predictions non implémenté).")
