import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")
MLFLOW_INTERNAL_URL = os.environ.get("MLFLOW_INTERNAL_URL", MLFLOW_URL)
API_DOCS_URL = os.environ.get("API_DOCS_URL", "http://localhost:8000/docs")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_INTERNAL_URL = os.environ.get("AIRFLOW_INTERNAL_URL", AIRFLOW_URL)

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

_PRESET_DEFAULT = {
    "loan_amnt": 10000, "int_rate": 12.0, "installment": 300,
    "annual_inc": 60000, "dti": 15.0, "delinq_2yrs": 0,
    "fico_range": (690, 694), "inq_last_6mths": 0, "open_acc": 10,
    "pub_rec": 0, "revol_bal": 15000, "revol_util": 50.0, "total_acc": 20,
    "term": " 36 months", "grade": "C", "sub_grade": "C3",
    "emp_length": "5 years", "home_ownership": "RENT",
    "verification_status": "Not Verified", "purpose": "debt_consolidation",
}
_PRESET_GOOD = {
    "loan_amnt": 8000, "int_rate": 7.0, "installment": 180,
    "annual_inc": 85000, "dti": 6.0, "delinq_2yrs": 0,
    "fico_range": (730, 734), "inq_last_6mths": 0, "open_acc": 8,
    "pub_rec": 0, "revol_bal": 4000, "revol_util": 15.0, "total_acc": 14,
    "term": " 36 months", "grade": "A", "sub_grade": "A2",
    "emp_length": "5 years", "home_ownership": "MORTGAGE",
    "verification_status": "Verified", "purpose": "debt_consolidation",
}
_PRESET_RISKY = {
    "loan_amnt": 28000, "int_rate": 25.0, "installment": 850,
    "annual_inc": 32000, "dti": 32.0, "delinq_2yrs": 3,
    "fico_range": (590, 594), "inq_last_6mths": 6, "open_acc": 22,
    "pub_rec": 2, "revol_bal": 42000, "revol_util": 88.0, "total_acc": 32,
    "term": " 60 months", "grade": "F", "sub_grade": "F3",
    "emp_length": "< 1 year", "home_ownership": "RENT",
    "verification_status": "Not Verified", "purpose": "small_business",
}

for key, val in _PRESET_DEFAULT.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Page background ── */
    .stApp { background-color: #F8FAFC; }

    /* ── Card title ── */
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: .06em;
        color: #64748B;
        margin-bottom: .75rem;
        display: flex;
        align-items: center;
        gap: .4rem;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #1E3A5F;
        padding: .5rem 0 .25rem 0;
        border-bottom: 2px solid #2563EB;
        margin-bottom: 1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-logo {
        font-size: 1.3rem;
        font-weight: 800;
        color: #2563EB;
        letter-spacing: -.02em;
        margin-bottom: .25rem;
    }
    .sidebar-sub {
        font-size: .75rem;
        color: #94A3B8;
        margin-bottom: 1.25rem;
    }
    .service-badge {
        display: flex;
        align-items: center;
        gap: .5rem;
        padding: .5rem .75rem;
        border-radius: 8px;
        margin-bottom: .4rem;
        font-size: .85rem;
        font-weight: 500;
        background: #F1F5F9;
        color: #1E293B;
        text-decoration: none;
    }
    .dot-green { color: #16A34A; font-size: 1.1rem; }
    .dot-gray  { color: #94A3B8; font-size: 1.1rem; }

    /* ── Preset buttons ── */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: .75rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }

    /* ── Submit button ── */
    div[data-testid="stFormSubmitButton"] > button {
        background: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: .65rem 0 !important;
        font-size: 1rem !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1D4ED8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _risk_band(prob: float) -> tuple[str, str, str]:
    if prob < 0.30:
        return "Faible", "#16A34A", "✅"
    if prob < 0.60:
        return "Modéré", "#D97706", "⚠️"
    return "Élevé", "#DC2626", "🔴"


def _ping(url: str) -> bool:
    try:
        httpx.get(url, timeout=1.5).raise_for_status()
        return True
    except Exception:
        return False


def _gauge(prob: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 48, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar": {"color": color},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#DCFCE7"},
                    {"range": [30, 60], "color": "#FEF9C3"},
                    {"range": [60, 100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": round(prob * 100, 1),
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin={"t": 20, "b": 0, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "sans-serif"},
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">LendingClub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Prédiction de défaut de paiement</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Services**")
    services = [
        ("API", API_URL + "/health", API_DOCS_URL),
        ("MLflow", MLFLOW_INTERNAL_URL + "/health", MLFLOW_URL),
        ("Airflow", AIRFLOW_INTERNAL_URL + "/airflow/health", AIRFLOW_URL),
    ]
    for name, ping_url, link_url in services:
        up = _ping(ping_url)
        dot_class = "dot-green" if up else "dot-gray"
        dot = "●"
        status = "en ligne" if up else "hors ligne"
        st.markdown(
            f'<a href="{link_url}" target="_blank" class="service-badge">'
            f'<span class="{dot_class}">{dot}</span> {name} — <span style="color:#94A3B8;font-size:.8rem">{status}</span>'
            f"</a>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("Réalisé par **Théo ELOY**")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_home, tab_pred, tab_hist = st.tabs(["🏠 Accueil", "🔮 Prédiction", "📊 Historique"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB — ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown('<div class="section-header">Problématique métier</div>', unsafe_allow_html=True)
    st.markdown(
        """
        **LendingClub** est une plateforme américaine de prêt entre particuliers (*peer-to-peer lending*).
        Les emprunteurs soumettent une demande de prêt, et des investisseurs particuliers financent ces prêts
        en échange d'intérêts. Le risque principal pour l'investisseur est le **défaut de paiement** :
        l'emprunteur ne rembourse pas son prêt.

        L'enjeu est considérable : sur les données 2007–2018, environ **20 % des prêts accordés**
        se terminent en défaut (*charged off*), représentant des pertes significatives pour les investisseurs.

        **Objectif :** construire un modèle de classification binaire capable de prédire,
        à partir des caractéristiques d'un dossier de prêt, si l'emprunteur fera défaut.
        """
    )

    st.divider()
    st.markdown('<div class="section-header">Le dataset</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Prêts enregistrés", "2 260 668")
    col2.metric("Variables disponibles", "~150")
    col3.metric("Taux de défaut", "~20 %")

    st.markdown(
        """
        Les données proviennent du dataset public **Lending Club** (Kaggle), couvrant les prêts émis entre 2007 et 2018.

        | Variable | Description |
        |---|---|
        | `loan_amnt` | Montant du prêt demandé |
        | `int_rate` | Taux d'intérêt appliqué |
        | `fico_range_low/high` | Score de crédit FICO |
        | `dti` | Ratio dette / revenu annuel |
        | `annual_inc` | Revenu annuel déclaré |
        | `grade` / `sub_grade` | Note de risque LendingClub |
        | `purpose` | Motif du prêt |
        | `delinq_2yrs` | Incidents de paiement sur 2 ans |
        """
    )

    st.divider()
    st.markdown(
        '<div class="section-header">Approche Machine Learning</div>', unsafe_allow_html=True
    )
    st.markdown(
        """
        1. **Préparation** — nettoyage, encodage, normalisation
        2. **Entraînement** — comparaison de 4 modèles en validation croisée 5-fold
        3. **Sélection** — meilleur AUC-ROC enregistré via **MLflow**
        4. **Serving** — API FastAPI + interface Streamlit
        """
    )

    model_data = {
        "Modèle": ["LightGBM", "Logistic Reg.", "XGBoost", "Random Forest"],
        "AUC-ROC": [0.689, 0.684, 0.671, 0.666],
    }
    df_models = pd.DataFrame(model_data).set_index("Modèle")

    fig_models = go.Figure(
        go.Bar(
            x=df_models.index.tolist(),
            y=df_models["AUC-ROC"].tolist(),
            marker_color=["#2563EB", "#60A5FA", "#93C5FD", "#BFDBFE"],
            text=[f"{v:.3f}" for v in df_models["AUC-ROC"]],
            textposition="outside",
        )
    )
    fig_models.update_layout(
        title="Comparaison des modèles (AUC-ROC, CV 5-fold)",
        yaxis={"range": [0.64, 0.70], "title": "AUC-ROC"},
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin={"t": 40, "b": 0, "l": 0, "r": 0},
        font={"family": "sans-serif", "color": "#1E293B"},
    )
    fig_models.update_yaxes(showgrid=True, gridcolor="#E2E8F0")
    st.plotly_chart(fig_models, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB — PRÉDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pred:

    # ── Résultat (affiché en haut si disponible) ───────────────────────────
    result_slot = st.empty()

    if "pred_result" in st.session_state:
        prob = st.session_state.pred_result["default_probability"]
        prediction = st.session_state.pred_result["prediction"]
        risk_label, risk_color, icon = _risk_band(prob)

        with result_slot.container():
            col_gauge, col_info = st.columns([1, 1])
            with col_gauge:
                st.plotly_chart(_gauge(prob, risk_color), use_container_width=True)
            with col_info:
                st.markdown(
                    f"""
                    <div style="display:flex;flex-direction:column;justify-content:center;
                                height:260px;padding:1rem;">
                        <div style="font-size:2.5rem;font-weight:800;color:{risk_color};
                                    line-height:1;">{icon} {risk_label}</div>
                        <div style="font-size:1rem;color:#64748B;margin:.5rem 0 1.25rem 0;">
                            Niveau de risque estimé
                        </div>
                        <div style="background:#F1F5F9;border-radius:8px;padding:.75rem 1rem;">
                            <div style="font-size:.8rem;color:#94A3B8;font-weight:600;
                                        text-transform:uppercase;letter-spacing:.05em;">
                                Classe prédite
                            </div>
                            <div style="font-size:1.4rem;font-weight:700;color:#1E293B;">
                                {prediction}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.divider()

        if st.session_state.pop("scroll_top", False):
            components.html(
                """
                <script>
                    (function() {
                        var selectors = [
                            '[data-testid="stMain"]',
                            '[data-testid="stAppViewContainer"]',
                            '.main',
                            'section.main'
                        ];
                        function tryScroll(attempt) {
                            for (var i = 0; i < selectors.length; i++) {
                                var el = window.parent.document.querySelector(selectors[i]);
                                if (el) { el.scrollTop = 0; return; }
                            }
                            if (attempt < 15) setTimeout(function() { tryScroll(attempt + 1); }, 80);
                        }
                        tryScroll(0);
                    })();
                </script>
                """,
                height=0,
            )

    # ── Presets ────────────────────────────────────────────────────────────
    st.caption("Charger un profil type :")
    p1, p2, p3 = st.columns(3)
    if p1.button("📋 Profil par défaut", use_container_width=True):
        for k, v in _PRESET_DEFAULT.items():
            st.session_state[k] = v
        st.session_state.pop("pred_result", None)
    if p2.button("✅ Bon emprunteur", use_container_width=True):
        for k, v in _PRESET_GOOD.items():
            st.session_state[k] = v
        st.session_state.pop("pred_result", None)
    if p3.button("⚠️ Emprunteur à risque", use_container_width=True):
        for k, v in _PRESET_RISKY.items():
            st.session_state[k] = v
        st.session_state.pop("pred_result", None)

    # ── Formulaire ─────────────────────────────────────────────────────────
    with st.form("prediction_form"):

        with st.container(border=True):
            st.markdown('<div class="card-title">💳 Caractéristiques du prêt</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                loan_amnt = st.slider("Montant ($)", 500, 40000, step=500, key="loan_amnt")
                int_rate = st.slider("Taux d'intérêt (%)", 5.0, 30.0, step=0.5, key="int_rate")
                installment = st.slider("Mensualité ($)", 10, 1500, step=10, key="installment")
            with c2:
                term = st.selectbox("Durée", _TERM, key="term")
                purpose = st.selectbox("Motif", _PURPOSE, key="purpose")
                grade = st.selectbox("Note de risque", _GRADE, key="grade")
                sub_grade = st.selectbox("Sous-note", _SUB_GRADE, key="sub_grade")

        with st.container(border=True):
            st.markdown('<div class="card-title">💰 Profil financier</div>', unsafe_allow_html=True)
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
                emp_length = st.selectbox("Ancienneté pro.", _EMP_LENGTH, key="emp_length")
                home_ownership = st.selectbox("Résidence", _HOME_OWNERSHIP, key="home_ownership")
                verification_status = st.selectbox(
                    "Vérification revenu", _VERIFICATION, key="verification_status"
                )

        with st.container(border=True):
            st.markdown('<div class="card-title">📁 Historique de crédit</div>', unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                delinq_2yrs = st.slider(
                    "Incidents de paiement (2 ans)", 0, 20, step=1, key="delinq_2yrs"
                )
                inq_last_6mths = st.slider(
                    "Demandes de crédit (6 mois)", 0, 15, step=1, key="inq_last_6mths"
                )
                pub_rec = st.slider("Mentions publiques", 0, 10, step=1, key="pub_rec")
            with c6:
                open_acc = st.slider("Comptes ouverts", 0, 60, step=1, key="open_acc")
                total_acc = st.slider("Total lignes de crédit", 1, 100, step=1, key="total_acc")

        submitted = st.form_submit_button("🔮 Lancer la prédiction", use_container_width=True)

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
            st.session_state["pred_result"] = resp.json()
            st.session_state["scroll_top"] = True
            st.rerun()
        except httpx.HTTPError as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB — HISTORIQUE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    if st.button("🔄 Rafraîchir", use_container_width=False):
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
                fig_dist = go.Figure(
                    go.Histogram(
                        x=df["default_probability"].tolist(),
                        nbinsx=10,
                        marker_color="#2563EB",
                        marker_line_width=0,
                        xbins={"start": 0, "end": 1, "size": 0.1},
                    )
                )
                fig_dist.update_layout(
                    title="Distribution des probabilités",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin={"t": 40, "b": 40, "l": 0, "r": 0},
                    font={"family": "sans-serif", "color": "#1E293B"},
                    xaxis={"title": "Probabilité de défaut", "tickformat": ".0%"},
                    yaxis={"title": "Nombre", "showgrid": True, "gridcolor": "#E2E8F0"},
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_chart2:
                pie_counts = df["prediction"].value_counts()
                fig_pie = go.Figure(
                    go.Pie(
                        labels=pie_counts.index.tolist(),
                        values=pie_counts.values.tolist(),
                        marker_colors=["#16A34A", "#DC2626"],
                        hole=0.45,
                        textinfo="label+percent",
                        textfont_size=13,
                    )
                )
                fig_pie.update_layout(
                    title="Répartition des prédictions",
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=300,
                    margin={"t": 40, "b": 0, "l": 0, "r": 0},
                    font={"family": "sans-serif", "color": "#1E293B"},
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

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
