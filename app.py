# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# -------------------- Page setup --------------------
st.set_page_config(page_title="Healthcare Claim Approval / Denial Predictor", layout="wide")
st.title("🏥 Healthcare Claim Approval / Denial Predictor")
st.caption(
    "Upload a cleaned CSV with columns such as Outcome or Claim Status (Paid/Denied), "
    "Reason Code, Insurance Type, Follow-up Required, AR Status, and amounts. "
    "The app will train a quick model and let you explore approval/denial likelihood."
)

# -------------------- Helpers --------------------
def _coalesce_outcome_cols(df: pd.DataFrame) -> pd.Series:
    """Return a single string outcome column from 'Outcome' or 'Claim Status'."""
    if "Outcome" in df.columns:
        s = df["Outcome"].astype(str).str.strip()
    elif "Claim Status" in df.columns:
        s = df["Claim Status"].astype(str).str.strip()
    else:
        raise ValueError("Expected a column named 'Outcome' or 'Claim Status' with values like 'Paid'/'Denied'.")
    return s

@st.cache_resource(show_spinner=False)
def load_dataframe(file_like_or_path):
    df = pd.read_csv(file_like_or_path)
    # standard tidy ups that won't alter schema meaningfully
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    outcome_str = _coalesce_outcome_cols(df)
    target = outcome_str.str.lower().map({"denied": 1, "paid": 0})
    if target.isna().all():
        raise ValueError("Could not map outcome values to 0/1. Expected values like 'Paid' and 'Denied'.")
    out = df.copy()
    out["Denied_Binary"] = target.fillna(0).astype(int)
    return out

def pick_feature_sets(df: pd.DataFrame):
    """Select common numeric and categorical features if present."""
    numeric = [c for c in ["Billed Amount", "Allowed Amount", "Paid Amount"] if c in df.columns]
    categorical = [c for c in [
        "Procedure Code", "Diagnosis Code", "Insurance Type",
        "Reason Code", "Follow-up Required", "AR Status"
    ] if c in df.columns]
    features = numeric + categorical
    if not features:
        raise ValueError(
            "No usable features found. Include at least some of: "
            "Billed/Allowed/Paid Amount, Reason Code, Insurance Type, Follow-up Required, AR Status, etc."
        )
    return numeric, categorical, features

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    # target
    df = make_target(df)
    y = df["Denied_Binary"]

    # cap outliers gently to help a simple linear model
    if "Billed Amount" in df.columns:
        cap = df["Billed Amount"].quantile(0.98)
        df["Billed Amount"] = np.minimum(df["Billed Amount"], cap)

    # features
    numeric, categorical, features = pick_feature_sets(df)
    X = df[features].copy()

    # preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop"
    )

    # model
    model = LogisticRegression(max_iter=500, class_weight="balanced")

    pipe = Pipeline(steps=[("prep", pre), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # metrics
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    cm = confusion_matrix(y_test, (proba >= 0.5).astype(int))

    # schema for UI
    schema = {
        "numeric": {c: (float(X[c].min()), float(X[c].max()), float(X[c].median())) for c in numeric},
        "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in categorical},
        "features": features
    }
    return pipe, (auc, acc, cm), schema, df

def sidebar_inputs(schema: dict) -> pd.DataFrame:
    st.sidebar.header("Set claim inputs")

    vals = {}
    # numeric
    for c, (_mn, _mx, med) in schema["numeric"].items():
        vals[c] = st.sidebar.number_input(c, min_value=0.0, value=float(med))

    # categorical
    for c, opts in schema["categorical"].items():
        if opts:
            vals[c] = st.sidebar.selectbox(c, opts, index=0)
        else:
            vals[c] = None

    return pd.DataFrame([vals])

def verdict(prob_denied: float):
    colA, colB = st.columns([1, 2])
    with colA:
        st.markdown("### ❌ Likely **Denied**" if prob_denied >= 0.5 else "### ✅ Likely **Approved**")
    with colB:
        st.caption("Estimated denial probability")
        st.progress(float(np.clip(prob_denied, 0.0, 1.0)))
        st.caption(f"{prob_denied:.1%}")

def hint_panel(row: pd.Series, schema: dict):
    st.subheader("Why this result? (quick hints)")
    notes = []
    for c, (_mn, _mx, med) in schema["numeric"].items():
        v = row[c]
        if med != 0:
            rel = (v - med) / abs(med)
            if abs(rel) > 0.15:
                notes.append(f"- **{c}** is {'above' if rel > 0 else 'below'} typical levels")
    for c in schema["categorical"].keys():
        notes.append(f"- **{c}** = **{row[c]}**")
    st.markdown("\n".join(notes) or "_Inputs are near typical dataset values._")

def small_cards(df: pd.DataFrame, auc: float):
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Rows", f"{len(df):,}")
    with k2:
        st.metric("Denial rate (dataset)", f"{df['Denied_Binary'].mean():.0%}")
    with k3:
        st.metric("Validation AUC", f"{auc:.3f}")

def quick_charts(df: pd.DataFrame):
    st.subheader("Quick denial insights")
    left, mid, right = st.columns(3)

    def rate_bar(ax, series, label):
        top = series.value_counts().head(5).index
        tmp = df[df[series.name].isin(top)].copy()
        tmp["Is_Denied"] = df["Denied_Binary"]
        tmp = tmp.groupby(series.name)["Is_Denied"].mean().reset_index()
        ax.barh(tmp[series.name], tmp["Is_Denied"])
        ax.set_xlabel("Denial rate")
        ax.set_title(label)
        ax.invert_yaxis()

    if "Reason Code" in df.columns:
        with left:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            rate_bar(ax, df["Reason Code"], "By Reason Code")
            st.pyplot(fig, clear_figure=True)

    if "Insurance Type" in df.columns:
        with mid:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            rate_bar(ax, df["Insurance Type"], "By Insurance Type")
            st.pyplot(fig, clear_figure=True)

    if "Follow-up Required" in df.columns:
        with right:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            rate_bar(ax, df["Follow-up Required"], "By Follow-up Required")
            st.pyplot(fig, clear_figure=True)

# -------------------- App flow --------------------
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
use_local = st.toggle("Use local path instead", value=False)
local_path = st.text_input("Local CSV path", value="cleaned_claim_data_final.csv")

df = None
try:
    if uploaded is not None:
        df = load_dataframe(uploaded)
    elif use_local:
        df = load_dataframe(local_path)
except Exception as e:
    st.error(f"Failed to read file: {e}")

if df is None:
    st.info("👈 Upload a CSV or toggle 'Use local path instead' to continue.")
    st.stop()

try:
    pipe, (auc, acc, cm), schema, df2 = train_model(df)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

# dataset summary
small_cards(df2, auc)
st.divider()

# charts
quick_charts(df2)

st.divider()
st.subheader("Interactive prediction")

# sidebar inputs
user_df = sidebar_inputs(schema)

# predict
if st.sidebar.button("Predict"):
    try:
        p_denied = float(pipe.predict_proba(user_df)[0, 1])
        verdict(p_denied)
        hint_panel(user_df.iloc[0], schema)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Set inputs in the sidebar and click Predict.")
