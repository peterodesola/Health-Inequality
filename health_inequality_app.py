import os
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Gender Inequality Dashboard",
    page_icon="📊",
    layout="wide"
)

plt.style.use("seaborn-v0_8")

DATA_PATH = r"gender_inequality_index.csv"
TARGET_COL = "GII VALUE"

NUMERIC_COLS = [
    "HDI rank", "GII VALUE", "GII RANK",
    "Maternal_mortality", "Adolescent_birth_rate",
    "Seats_parliamentt(% held by women)",
    "F_secondary_educ", "M_secondary_educ",
    "F_Labour_force", "M_Labour_force"
]


# ================= HELPER FUNCTIONS =================
@st.cache_data
def load_data(path: str):
    """Load and lightly clean the gender inequality dataset."""
    if not os.path.exists(path):
        return None, f"File not found at: {path}"

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        return None, f"Error reading CSV: {e}"

    # Replace placeholder missing markers
    df = df.replace("..", np.nan)

    # Standardise column names (strip spaces/BOM)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()

    # Convert numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering if columns exist
    if "F_secondary_educ" in df.columns and "M_secondary_educ" in df.columns:
        df["Edu_gap"] = df["M_secondary_educ"] - df["F_secondary_educ"]
    else:
        df["Edu_gap"] = np.nan

    if "F_Labour_force" in df.columns and "M_Labour_force" in df.columns:
        df["Labour_gap"] = df["M_Labour_force"] - df["F_Labour_force"]
    else:
        df["Labour_gap"] = np.nan

    return df, None


def safe_numeric_summary(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T


# ================= LOAD DATA =================
df, load_error = load_data(DATA_PATH)

if load_error:
    st.error(load_error)
    st.stop()

if df is None or df.empty:
    st.error("Dataset is empty or could not be loaded.")
    st.stop()


# ================= INTRO & PROBLEM STATEMENT =================
st.title("📊 Global Gender Inequality Dashboard")

st.markdown(
    """
This dashboard explores global **gender inequality** using a country‑level dataset that
includes the Gender Inequality Index (GII), maternal health, adolescent birth rates,
education, labour‑force participation and women’s political representation.

The main questions are:

- How does gender inequality vary across human development groups and regions?
- How is GII related to reproductive health, education and labour‑force gaps?
- Which countries are performing better or worse on key gender‑equality indicators?

Use the sections below to review key performance indicators, visualise patterns
and read a brief summary of findings.
"""
)

st.markdown("---")


# ================= KPI SECTION =================
st.header("Key Performance Indicators")

df_gii = df.dropna(subset=[TARGET_COL])

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_countries = int(df_gii["Country"].nunique()) if "Country" in df_gii.columns else len(df_gii)
    st.metric("Number of countries", n_countries)

with col2:
    st.metric("Mean GII value", f"{df_gii[TARGET_COL].mean():.3f}")

with col3:
    st.metric("Lowest GII value", f"{df_gii[TARGET_COL].min():.3f}")

with col4:
    st.metric("Highest GII value", f"{df_gii[TARGET_COL].max():.3f}")

st.markdown("---")


# ================= LAYOUT: VISUALS =================
tab_dist, tab_hd, tab_rel, tab_map, tab_summary = st.tabs(
    ["Distribution", "By development group", "Relationships", "Global map", "Findings & context"]
)


# ---------- Distribution ----------
with tab_dist:
    st.subheader("Distribution of Gender Inequality Index")

    if TARGET_COL in df_gii.columns:
        fig_hist = px.histogram(
            df_gii,
            x=TARGET_COL,
            nbins=20,
            title="GII distribution",
            labels={TARGET_COL: "GII VALUE"}
        )
        fig_hist.update_traces(marker_color="#4472c4", opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("**Summary statistics (GII VALUE and key indicators)**")
        summary_cols = [
            "HDI rank", "GII VALUE", "GII RANK",
            "Maternal_mortality", "Adolescent_birth_rate",
            "Seats_parliamentt(% held by women)",
            "F_secondary_educ", "M_secondary_educ",
            "F_Labour_force", "M_Labour_force",
            "Edu_gap", "Labour_gap",
        ]
        desc = safe_numeric_summary(df, summary_cols)
        if not desc.empty:
            st.dataframe(desc.style.format("{:.3f}"))
        else:
            st.info("No numeric columns available for summary.")
    else:
        st.warning("GII VALUE column is missing; cannot show distribution.")


# ---------- By human development ----------
with tab_hd:
    st.subheader("GII by Human Development group")

    if "HUMAN DEVELOPMENT" in df.columns and TARGET_COL in df.columns:
        df_box = df.dropna(subset=["HUMAN DEVELOPMENT", TARGET_COL])
        if df_box.empty:
            st.info("No complete data for HUMAN DEVELOPMENT and GII VALUE.")
        else:
            order = ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]
            order = [o for o in order if o in df_box["HUMAN DEVELOPMENT"].unique()]
            fig_box = px.box(
                df_box,
                x="HUMAN DEVELOPMENT",
                y=TARGET_COL,
                category_orders={"HUMAN DEVELOPMENT": order},
                points="all",
                labels={"HUMAN DEVELOPMENT": "Human development group", TARGET_COL: "GII VALUE"},
                title="GII distribution by human development"
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("HUMAN DEVELOPMENT or GII VALUE column not found in the dataset.")


# ---------- Relationships ----------
with tab_rel:
    st.subheader("Relationships between GII and key drivers")

    # Selectors with safe defaults
    numeric_options = [
        c for c in [
            "Maternal_mortality", "Adolescent_birth_rate",
            "Seats_parliamentt(% held by women)",
            "F_secondary_educ", "M_secondary_educ",
            "F_Labour_force", "M_Labour_force",
            "Edu_gap", "Labour_gap"
        ]
        if c in df.columns
    ]

    if not numeric_options:
        st.info("No numeric driver columns available for relationship plots.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            x_var = st.selectbox("Select X‑axis variable", options=numeric_options, index=0)
        with col_b:
            color_by = st.selectbox(
                "Colour by (optional)",
                options=["None", "HUMAN DEVELOPMENT", "Country"] if "HUMAN DEVELOPMENT" in df.columns else ["None", "Country"],
                index=1 if "HUMAN DEVELOPMENT" in df.columns else 0
            )

        subset_cols = [TARGET_COL, x_var]
        if color_by != "None" and color_by in df.columns:
            subset_cols.append(color_by)

        df_scatter = df.dropna(subset=subset_cols)

        if df_scatter.empty:
            st.info("No complete rows available for the selected variables.")
        else:
            fig_scatter = px.scatter(
                df_scatter,
                x=x_var,
                y=TARGET_COL,
                color=color_by if color_by in df_scatter.columns else None,
                hover_name="Country" if "Country" in df_scatter.columns else None,
                trendline="ols",
                labels={TARGET_COL: "GII VALUE"},
                title=f"GII vs {x_var}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


# ---------- Global map ----------
with tab_map:
    st.subheader("Global map of Gender Inequality Index")

    if "Country" in df.columns and TARGET_COL in df.columns:
        df_map = df.dropna(subset=["Country", TARGET_COL])
        if df_map.empty:
            st.info("No complete Country + GII VALUE rows available for mapping.")
        else:
            try:
                fig_map = px.choropleth(
                    df_map,
                    locations="Country",
                    locationmode="country names",
                    color=TARGET_COL,
                    color_continuous_scale="RdYlGn_r",
                    hover_name="Country",
                    hover_data={
                        TARGET_COL: ":.3f",
                        "Maternal_mortality": True if "Maternal_mortality" in df_map.columns else False,
                        "Adolescent_birth_rate": True if "Adolescent_birth_rate" in df_map.columns else False
                    },
                    title="Global map of Gender Inequality Index"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render map because of location name issues: {e}")
    else:
        st.info("Country or GII VALUE column missing; cannot render map.")


# ---------- Findings & problem statement ----------
with tab_summary:
    st.subheader("Problem statement")

    st.markdown(
        """
Many countries still experience substantial gender‑based inequalities in health,
education, political representation and labour‑force participation. Decision‑makers
need clear, data‑driven tools to understand **where** inequality is highest and
**which factors** are most strongly associated with it.
"""
    )

    st.subheader("Key findings (from this dataset)")

    st.markdown(
        """
- **Development gradient:** Countries classified as VERY HIGH human development
  tend to have the lowest GII values, while MEDIUM and LOW categories show
  systematically higher inequality.
- **Reproductive health:** Higher maternal mortality and adolescent birth rates
  are associated with higher GII, indicating that reproductive health remains a
  core driver of gender inequality.
- **Representation and education:** Higher shares of seats held by women in
  parliament, and smaller gaps between male and female secondary education,
  align with lower GII values.
- **Labour‑force gaps:** Countries where women’s labour‑force participation
  is much lower than men’s tend to have higher GII, although the relationship
  is weaker than for health and education.
"""
    )

    st.subheader("How to use this dashboard")

    st.markdown(
        """
- Start with the **KPIs** to get a global picture of gender inequality levels.
- Use **Distribution** to see how GII values are spread across countries.
- Explore **By development group** to compare inequality across human
  development categories.
- Use **Relationships** to investigate how GII relates to specific drivers
  such as maternal mortality, adolescent births, education and labour gaps.
- View the **Global map** to identify regional patterns and clusters of
  high or low gender inequality.
"""
    )
