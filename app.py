import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="DevScope | Developer Survey Insights",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME / UI --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root { color-scheme: dark; }
html, body, [class*="css"], [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    color: #edf2ff !important;
}
.stApp { background: #0b1020; color: #edf2ff !important; }
[data-testid="stHeader"] { background: #0b1020 !important; }
[data-testid="stToolbar"] { color: #edf2ff !important; }
[data-testid="stSidebar"] {
    background: #111a30 !important;
    border-right: 1px solid #253452;
}
[data-testid="stSidebar"] * {
    color: #e6edff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #253047 !important;
}
[data-testid="stSidebar"] input {
    color: #253047 !important;
}
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif;
    color: #f4f7ff !important;
}
p, li, label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
    color: #dce6f8 !important;
}
[data-testid="stMetric"] {
    background: #141f36;
    border: 1px solid #293b5d;
    padding: 16px 18px;
    border-radius: 16px;
}
[data-testid="stMetricLabel"] { color: #a9bbd8 !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
[data-testid="stMetricDelta"] { color: #9fd9ff !important; }
[data-testid="stRadio"] label, [data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] label { color: #e6edff !important; }
.hero {
    padding: 26px 30px; border-radius: 22px;
    background: linear-gradient(115deg, #172b4d 0%, #243e70 55%, #3c3274 100%);
    border: 1px solid #405987; margin-bottom: 22px;
}
.hero-kicker { color: #9fd9ff !important; font-size: 12px; font-weight: 700; letter-spacing: 2px; }
.hero h1 { font-size: 34px; margin: 8px 0 5px 0; color: #ffffff !important; }
.hero p { color: #e0eaff !important; margin: 0; font-size: 15px; }
.section-note { color: #aab9d3 !important; font-size: 13px; }
.insight {
    background: #142541; border-left: 4px solid #54c5d8;
    border-radius: 10px; padding: 12px 16px; color: #dce9ff !important;
}
.small-muted { color: #9aabc8 !important; font-size: 12px; }
[data-testid="stDataFrame"] { color: #253047 !important; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "survey_cleaned.csv.gz"

@st.cache_data(show_spinner="Loading developer survey...")
def load_data(path):
    data = pd.read_csv(path, low_memory=False)
    return data

def split_multi(series):
    """Turn semicolon-separated survey answers into one answer per row."""
    return (series.dropna().astype(str).str.split(";").explode().str.strip()
            .replace("", pd.NA).dropna())

def parse_experience(series):
    """Convert survey experience answers to numeric years."""
    s = series.astype("string").str.strip()
    s = s.replace({
        "Less than 1 year": "0.5",
        "More than 50 years": "51"
    })
    return pd.to_numeric(s, errors="coerce")

def make_bar(data, x, y, title, horizontal=False):
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation="h", title=title,
                     color=y, color_continuous_scale="Tealgrn")
    else:
        fig = px.bar(data, x=x, y=y, title=title,
                     color=y, color_continuous_scale="Tealgrn")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=55,b=10),
        coloraxis_showscale=False, height=390
    )
    return fig

# -------------------- LOAD DATA --------------------
if not DATA_PATH.exists():
    st.error("Dataset not found. Ensure data/survey_cleaned.csv.gz is included in the GitHub repository.")
    st.stop()

df = load_data(str(DATA_PATH))

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.markdown("## 💻 DevScope")
st.sidebar.caption("Developer survey intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Explore dashboard",
    ["Overview", "Technology Trends", "Global Developer Map", "Salary & Experience",
     "Education & Employment", "Insights & Data"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

countries = sorted(df["Country"].dropna().unique().tolist())
country_options = st.sidebar.multiselect("Country", countries, placeholder="All countries")
employment_options = sorted(df["Employment"].dropna().unique().tolist())
employment_selected = st.sidebar.multiselect("Employment status", employment_options,
                                              placeholder="All employment types")

filtered = df.copy()
if country_options:
    filtered = filtered[filtered["Country"].isin(country_options)]
if employment_selected:
    filtered = filtered[filtered["Employment"].isin(employment_selected)]

st.sidebar.caption(f"Showing {len(filtered):,} of {len(df):,} responses")
st.sidebar.caption("Source: Stack Overflow Developer Survey 2022")

# -------------------- HEADER / KPI --------------------
st.markdown("""
<div class="hero">
  <div class="hero-kicker">DATA • TECHNOLOGY • CAREERS</div>
  <h1>Developer Career & Technology Trends</h1>
  <p>Explore programming skills, technology interests, global participation,
  compensation patterns and developer career pathways.</p>
</div>
""", unsafe_allow_html=True)

total = len(filtered)
country_count = filtered["Country"].nunique()
lang_count = len(split_multi(filtered["LanguageHaveWorkedWith"])) if "LanguageHaveWorkedWith" in filtered else 0
salary_values = pd.to_numeric(filtered["ConvertedCompYearly"], errors="coerce").dropna()
salary_values = salary_values[salary_values > 0]
median_salary = salary_values.median() if not salary_values.empty else float("nan")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Survey responses", f"{total:,}")
k2.metric("Countries represented", f"{country_count:,}")
k3.metric("Language selections", f"{lang_count:,}")
k4.metric("Median reported compensation", f"${median_salary:,.0f}" if pd.notna(median_salary) else "N/A")

st.markdown("")

# -------------------- PAGES --------------------
if page == "Overview":
    st.subheader("At a glance")
    a, b = st.columns([1.15, 1])
    with a:
        top_countries = filtered["Country"].value_counts().head(10).rename_axis("Country").reset_index(name="Responses")
        st.plotly_chart(make_bar(top_countries, "Country", "Responses", "Top 10 countries", True),
                        use_container_width=True)
    with b:
        emp = filtered["Employment"].dropna().value_counts().head(8).rename_axis("Employment").reset_index(name="Responses")
        fig = px.pie(emp, names="Employment", values="Responses", hole=0.58,
                     title="Employment mix", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10,r=10,t=55,b=10), height=390)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight">Use the sidebar filters to compare countries and employment groups. The figures update together.</div>',
                unsafe_allow_html=True)
    st.caption("Survey responses are self-reported and should not be interpreted as a complete census of all developers.")

elif page == "Technology Trends":
    st.subheader("Technology usage vs learning interest")
    categories = {
        "Programming languages": ("LanguageHaveWorkedWith", "LanguageWantToWorkWith"),
        "Databases": ("DatabaseHaveWorkedWith", "DatabaseWantToWorkWith"),
        "Web frameworks": ("WebframeHaveWorkedWith", "WebframeWantToWorkWith"),
        "Platforms": ("PlatformHaveWorkedWith", "PlatformWantToWorkWith"),
        "Tools & technologies": ("ToolsTechHaveWorkedWith", "ToolsTechWantToWorkWith"),
        "Miscellaneous technologies": ("MiscTechHaveWorkedWith", "MiscTechWantToWorkWith"),
    }
    category = st.selectbox("Technology category", list(categories.keys()))
    have_col, want_col = categories[category]
    left, right = st.columns(2)
    have = split_multi(filtered[have_col]).value_counts().head(15).rename_axis("Technology").reset_index(name="Responses")
    want = split_multi(filtered[want_col]).value_counts().head(15).rename_axis("Technology").reset_index(name="Responses")
    with left:
        st.plotly_chart(make_bar(have, "Technology", "Responses", "Most used technologies", True),
                        use_container_width=True)
    with right:
        st.plotly_chart(make_bar(want, "Technology", "Responses", "Most wanted technologies", True),
                        use_container_width=True)

    have_counts = split_multi(filtered[have_col]).value_counts()
    want_counts = split_multi(filtered[want_col]).value_counts()
    comparison = pd.concat([have_counts.rename("Currently used"), want_counts.rename("Want to learn")], axis=1).fillna(0)
    comparison["Interest gap"] = comparison["Want to learn"] - comparison["Currently used"]
    comparison = comparison.sort_values("Interest gap", ascending=False).head(15).reset_index(names="Technology")
    st.markdown("#### Technologies with the largest interest gap")
    st.caption("This compares counts of respondents selecting a technology as wanted versus currently used. It is not proof of year-over-year growth.")
    fig = px.bar(comparison.sort_values("Interest gap"), x="Interest gap", y="Technology",
                 orientation="h", color="Interest gap", color_continuous_scale="Teal")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=480, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

elif page == "Global Developer Map":
    st.subheader("Where survey respondents are based")
    country = filtered["Country"].value_counts().rename_axis("Country").reset_index(name="Responses")
    st.plotly_chart(make_bar(country.head(20), "Country", "Responses", "Top 20 countries by response count", True),
                    use_container_width=True)
    st.dataframe(country.head(30), use_container_width=True, hide_index=True)
    st.caption("Counts represent survey respondents, not the total developer population in each country.")

elif page == "Salary & Experience":
    st.subheader("How professional experience relates to compensation")
    salary = filtered[["YearsCodePro", "ConvertedCompYearly", "Country", "DevType"]].copy()
    salary["Experience years"] = parse_experience(salary["YearsCodePro"])
    salary["Annual compensation (USD)"] = pd.to_numeric(salary["ConvertedCompYearly"], errors="coerce")
    salary = salary.dropna(subset=["Experience years", "Annual compensation (USD)"])
    salary = salary[(salary["Annual compensation (USD)"] > 0) &
                    (salary["Annual compensation (USD)"] <= salary["Annual compensation (USD)"].quantile(0.99))]
    if salary.empty:
        st.info("No salary and experience records match the current filters.")
    else:
        salary["Experience band"] = pd.cut(
            salary["Experience years"],
            bins=[-0.01, 1, 3, 5, 10, 15, 20, 30, 60],
            labels=["<1 year", "1–3", "3–5", "5–10", "10–15", "15–20", "20–30", "30+ years"]
        )
        grouped = salary.groupby("Experience band", observed=False)["Annual compensation (USD)"].median().dropna().reset_index()
        grouped.columns = ["Experience band", "Median annual compensation"]
        fig = px.line(grouped, x="Experience band", y="Median annual compensation",
                      markers=True, title="Median annual compensation by professional experience")
        fig.update_traces(line=dict(width=3), marker=dict(size=9))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Explore the distribution")
        fig2 = px.box(salary, x="Experience band", y="Annual compensation (USD)",
                      title="Compensation distribution by experience band", points=False)
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", height=430)
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Compensation is self-reported and converted to USD by the survey. Values above the 99th percentile are excluded from these charts to reduce extreme-value distortion. This is an association, not proof that experience causes higher pay.")

elif page == "Education & Employment":
    st.subheader("Education level and employment patterns")
    # Employment is a multi-select survey question: one response can contain
    # several statuses separated by semicolons. Split it before cross-tabulating.
    cross = filtered.dropna(subset=["EdLevel", "Employment"])[["EdLevel", "Employment"]].copy()
    cross["Employment"] = cross["Employment"].astype(str).str.split(";")
    cross = cross.explode("Employment")
    cross["Employment"] = cross["Employment"].str.strip()
    cross = cross[cross["Employment"].ne("")]
    if cross.empty:
        st.info("No education and employment records match the current filters.")
    else:
        table = pd.crosstab(cross["EdLevel"], cross["Employment"], normalize="index") * 100
        # Keep the chart readable by showing the most frequent employment categories.
        top_statuses = cross["Employment"].value_counts().head(9).index
        table_chart = table.reindex(columns=top_statuses, fill_value=0)
        fig = px.imshow(
            table_chart,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="Teal",
            labels=dict(x="Employment status", y="Education level", color="% within education")
        )
        fig.update_xaxes(tickangle=-25, tickfont=dict(size=10), side="bottom")
        fig.update_yaxes(tickfont=dict(size=11), autorange="reversed")
        fig.update_layout(
            title="Employment composition within each education group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edff"),
            height=560,
            margin=dict(l=20, r=20, t=65, b=150)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Employment is a multi-select survey question. Responses are split into individual statuses before calculating percentages. Each education row is normalized to 100% across the selected employment categories.")
        st.dataframe(table_chart.round(1), use_container_width=True)

elif page == "Insights & Data":
    st.subheader("Key observations and data quality")
    st.markdown("""
    - **Most popular skills:** identify the technologies with the highest number of current-use selections.
    - **Learning interest:** compare wanted technologies with current-use selections.
    - **Geography:** country rankings describe this survey's respondent distribution only.
    - **Salary:** compare medians and distributions rather than relying only on averages.
    - **Education:** use row percentages to compare employment composition across education groups.
    """)
    missing = filtered.isna().mean().mul(100).sort_values(ascending=False).head(15).rename("Missing (%)").reset_index()
    missing.columns = ["Column", "Missing (%)"]
    fig = px.bar(missing.sort_values("Missing (%)"), x="Missing (%)", y="Column",
                 orientation="h", title="Columns with the most missing values")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Download filtered data")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download filtered CSV", data=csv,
                       file_name="developer_survey_filtered.csv", mime="text/csv")
    st.markdown("#### Preview")
    st.dataframe(filtered.head(100), use_container_width=True)
    st.caption(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns. Current filtered shape: {filtered.shape[0]:,} × {filtered.shape[1]}.")

st.markdown("---")
st.markdown('<p class="small-muted">DevScope • Built with Python, Pandas, Streamlit and Plotly • Dataset: Stack Overflow Developer Survey 2022</p>',
            unsafe_allow_html=True)
