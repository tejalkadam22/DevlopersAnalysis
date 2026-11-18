import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

FONT_COLOR = "#e6edff"
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
        coloraxis_showscale=False, height=390,
        font=dict(color=FONT_COLOR, size=13), title_font=dict(color="#ffffff", size=18)
    )
    return fig


FONT_COLOR = "#e6edff"

def style_fig(fig, height=420, **layout):
    """Apply the dashboard theme and force light text so titles/legends stay readable."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT_COLOR, size=13), title_font=dict(color="#ffffff", size=18),
        legend=dict(font=dict(color=FONT_COLOR)), height=height, **layout
    )
    return fig

EMPLOYMENT_SHORT = {
    "Independent contractor, freelancer, or self-employed": "Freelancer / self-employed",
    "Employed, full-time": "Full-time employee",
    "Employed, part-time": "Part-time employee",
    "Student, full-time": "Full-time student",
    "Student, part-time": "Part-time student",
    "Not employed, but looking for work": "Job seeker",
    "Not employed, and not looking for work": "Not looking for work",
    "Retired": "Retired",
}

def short_employment(label):
    parts = [EMPLOYMENT_SHORT.get(p.strip(), p.strip()) for p in str(label).split(";")]
    return " + ".join(parts)

def make_employment_donut(series, top_n=6):
    """Readable donut: short names, small slices merged into 'Other', legend underneath."""
    counts = series.dropna().map(short_employment).value_counts()
    top = counts.head(top_n)
    other = counts.iloc[top_n:].sum()
    if other > 0:
        top = pd.concat([top, pd.Series({"Other combinations": other})])
    emp = top.rename_axis("Employment").reset_index(name="Responses")
    palette = ["#4cc9f0", "#f72585", "#ffd166", "#06d6a0", "#9b5de5", "#ff9f1c", "#8d99ae"]
    fig = px.pie(emp, names="Employment", values="Responses", hole=0.55,
                 color_discrete_sequence=palette)
    fig.update_traces(
        textinfo="percent", textposition="inside", insidetextorientation="horizontal",
        textfont=dict(size=13, color="#0b1020"), sort=False,
        marker=dict(line=dict(color="#0b1020", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value:,} responses (%{percent})<extra></extra>",
    )
    total = int(emp["Responses"].sum())
    fig.update_layout(
        title="Employment mix",
        uniformtext=dict(minsize=11, mode="hide"),   # tiny slices hide their label (no overlap)
        legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5,
                    font=dict(size=12, color=FONT_COLOR)),
        margin=dict(l=10, r=10, t=55, b=10),
        annotations=[dict(text=f"<b>{total:,}</b><br>responses", x=0.5, y=0.5,
                          font=dict(size=15, color="#ffffff"), showarrow=False)],
    )
    return style_fig(fig, height=480)

# ---------- Apriori (implemented from scratch, no extra library needed) ----------
APRIORI_SOURCES = {
    "Programming languages": "LanguageHaveWorkedWith",
    "Databases": "DatabaseHaveWorkedWith",
    "Web frameworks": "WebframeHaveWorkedWith",
    "Platforms": "PlatformHaveWorkedWith",
    "Tools & technologies": "ToolsTechHaveWorkedWith",
    "Miscellaneous technologies": "MiscTechHaveWorkedWith",
    "Developer roles": "DevType",
    "Learning interest: languages (want to work with)": "LanguageWantToWorkWith",
}

@st.cache_data(show_spinner="Running Apriori...")
def run_apriori(series, min_support, min_conf, max_len, top_items):
    """Classic level-wise Apriori. Returns (frequent itemsets df, rules df, n_transactions)."""
    trans = series.dropna().astype(str).str.split(";").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    trans = trans[trans.str.len() > 0]
    n = len(trans)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), 0
    exploded = trans.explode()
    items = exploded.value_counts().head(top_items).index.tolist()
    idx = {it: i for i, it in enumerate(items)}
    X = np.zeros((n, len(items)), dtype=bool)
    for r, lst in enumerate(trans.tolist()):
        for it in lst:
            j = idx.get(it)
            if j is not None:
                X[r, j] = True

    min_count = int(np.ceil(min_support * n))
    support = {}                       # frozenset(item names) -> count
    counts1 = X.sum(axis=0)
    current = []
    for j, c in enumerate(counts1):
        if c >= min_count:
            support[(j,)] = int(c)
            current.append((j,))
    k = 2
    while current and k <= max_len:
        prev = set(current)
        cands = set()
        for a, b in combinations(current, 2):
            if a[:-1] == b[:-1]:                    # join step
                cand = a + (b[-1],)
                if all(sub in prev for sub in combinations(cand, k - 1)):   # prune step
                    cands.add(cand)
        nxt = []
        for cand in sorted(cands)[:6000]:
            c = int(X[:, list(cand)].all(axis=1).sum())
            if c >= min_count:
                support[cand] = c
                nxt.append(cand)
        current = nxt
        k += 1

    fi = pd.DataFrame([{"Itemset": " + ".join(items[i] for i in s), "Size": len(s),
                        "Support": c / n, "Count": c} for s, c in support.items()])
    rules = []
    for s, c in support.items():
        if len(s) < 2:
            continue
        for r in range(1, len(s)):
            for ant in combinations(s, r):
                cons = tuple(i for i in s if i not in ant)
                conf = c / support[ant]
                if conf >= min_conf:
                    lift = conf / (support[cons] / n)
                    rules.append({"If a developer uses (antecedent)": " + ".join(items[i] for i in ant),
                                  "Then they also use (consequent)": " + ".join(items[i] for i in cons),
                                  "Support": c / n, "Confidence": conf, "Lift": lift,
                                  "Leverage": c / n - (support[ant] / n) * (support[cons] / n),
                                  "Count": c})
    rules = pd.DataFrame(rules)
    if not rules.empty:
        rules = rules.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    if not fi.empty:
        fi = fi.sort_values("Support", ascending=False).reset_index(drop=True)
    return fi, rules, n

# ---------- Regression helpers ----------
ED_MAP = [("Primary", 1), ("Secondary", 2), ("Some college", 3), ("Associate", 4),
          ("Bachelor", 5), ("Master", 6), ("Professional degree", 7), ("doctoral", 8)]
ORG_MAP = {"Just me": 1, "2 to 9": 2, "10 to 19": 3, "20 to 99": 4, "100 to 499": 5,
           "500 to 999": 6, "1,000 to 4,999": 7, "5,000 to 9,999": 8, "10,000 or more": 9}
AGE_MAP = {"Under 18": 16, "18-24": 21, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5,
           "55-64": 59.5, "65 years": 70}

def _map_contains(series, mapping):
    out = pd.Series(np.nan, index=series.index, dtype=float)
    s = series.astype("string")
    for key, val in mapping.items():
        out[s.str.contains(key, regex=False, na=False)] = val
    return out

@st.cache_data(show_spinner="Preparing regression data...")
def build_regression_data(frame):
    d = pd.DataFrame(index=frame.index)
    d["Salary"] = pd.to_numeric(frame["ConvertedCompYearly"], errors="coerce")
    d["Years of professional coding"] = parse_experience(frame["YearsCodePro"])
    d["Years of coding (total)"] = parse_experience(frame["YearsCode"])
    d["Total work experience"] = pd.to_numeric(frame["WorkExp"], errors="coerce")
    d["Age (midpoint)"] = _map_contains(frame["Age"], AGE_MAP)
    ed = pd.Series(np.nan, index=frame.index, dtype=float)
    s = frame["EdLevel"].astype("string")
    for key, val in ED_MAP:
        ed[s.str.contains(key, regex=False, na=False)] = val
    d["Education level (1-8)"] = ed
    d["Company size (1-9)"] = _map_contains(frame["OrgSize"], ORG_MAP)
    d["Lives in USA"] = (frame["Country"] == "United States of America").astype(float)
    rw = frame["RemoteWork"].astype("string")
    d["Fully remote"] = np.where(rw.isna(), np.nan, rw.str.contains("Fully remote", na=False).astype(float))
    d = d[d["Salary"] > 0]
    d = d[d["Salary"] <= d["Salary"].quantile(0.95)]
    return d

REG_FEATURES = ["Years of professional coding", "Years of coding (total)", "Total work experience",
                "Age (midpoint)", "Education level (1-8)", "Company size (1-9)",
                "Lives in USA", "Fully remote"]

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
     "Education & Employment", "Apriori Association Rules", "Regression Analysis",
     "Insights & Data"]
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
    a, b = st.columns([1.1, 1])
    with a:
        top_countries = filtered["Country"].value_counts().head(10).rename_axis("Country").reset_index(name="Responses")
        st.plotly_chart(make_bar(top_countries, "Country", "Responses", "Top 10 countries", True),
                        use_container_width=True)
    with b:
        if filtered["Employment"].dropna().empty:
            st.info("No employment data for the current filters.")
        else:
            st.plotly_chart(make_employment_donut(filtered["Employment"]), use_container_width=True)
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

elif page == "Apriori Association Rules":
    st.subheader("Apriori association rule mining")
    st.markdown(
        '<div class="insight"><b>What is Apriori?</b> An association-rule algorithm that finds items that '
        'frequently occur together. Here every respondent is one <i>transaction</i> and the technologies they '
        'selected are the <i>items</i>. A rule such as <i>Python → SQL</i> means developers who use Python '
        'often also use SQL.</div>', unsafe_allow_html=True)
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    source_label = c1.selectbox("Transactions (item type)", list(APRIORI_SOURCES.keys()))
    min_support = c2.slider("Minimum support", 0.01, 0.50, 0.05, 0.01,
                            help="Share of respondents that must contain the itemset.")
    min_conf = c3.slider("Minimum confidence", 0.10, 1.00, 0.60, 0.05,
                         help="P(consequent | antecedent).")
    max_len = c4.slider("Max itemset size", 2, 4, 3)
    min_lift = st.slider("Minimum lift (1 = no association, >1 = positive association)", 1.0, 5.0, 1.0, 0.1)

    fi, rules, n_trans = run_apriori(filtered[APRIORI_SOURCES[source_label]], min_support, min_conf, max_len, 40)
    if n_trans == 0:
        st.info("No transactions available for the current filters.")
    elif fi.empty:
        st.warning("No frequent itemsets found. Lower the minimum support.")
    else:
        rules_f = rules[rules["Lift"] >= min_lift] if not rules.empty else rules
        m1, m2, m3 = st.columns(3)
        m1.metric("Transactions (respondents)", f"{n_trans:,}")
        m2.metric("Frequent itemsets", f"{len(fi):,}")
        m3.metric("Association rules", f"{len(rules_f):,}")

        left, right = st.columns(2)
        with left:
            top_fi = fi.head(15).copy()
            fig = px.bar(top_fi.sort_values("Support"), x="Support", y="Itemset", orientation="h",
                         color="Support", color_continuous_scale="Tealgrn", title="Top frequent itemsets")
            fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(style_fig(fig, height=480), use_container_width=True)
        with right:
            if rules_f.empty:
                st.info("No rules meet the current confidence / lift thresholds. Try lowering them.")
            else:
                sc = rules_f.head(300).copy()
                sc["Rule"] = sc["If a developer uses (antecedent)"] + "  →  " + sc["Then they also use (consequent)"]
                fig = px.scatter(sc, x="Support", y="Confidence", size="Lift", color="Lift",
                                 hover_name="Rule", color_continuous_scale="Viridis",
                                 title="Rules: support vs confidence (size & colour = lift)")
                fig.update_xaxes(tickformat=".0%"); fig.update_yaxes(tickformat=".0%")
                fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(style_fig(fig, height=480), use_container_width=True)

        if not rules_f.empty:
            st.markdown("#### Strongest rules (sorted by lift)")
            show = rules_f.drop(columns=["Count"]).head(50).copy()
            top = show.iloc[0]
            st.markdown(
                f'<div class="insight">Top rule: developers using <b>{top.iloc[0]}</b> also use '
                f'<b>{top.iloc[1]}</b> in <b>{top["Confidence"]:.0%}</b> of cases '
                f'(support {top["Support"]:.1%}, lift {top["Lift"]:.2f}).</div>', unsafe_allow_html=True)
            st.dataframe(show, use_container_width=True, hide_index=True, column_config={
                "Support": st.column_config.NumberColumn(format="%.3f"),
                "Confidence": st.column_config.NumberColumn(format="%.3f"),
                "Lift": st.column_config.NumberColumn(format="%.2f"),
                "Leverage": st.column_config.NumberColumn(format="%.3f")})
        with st.expander("Frequent itemsets table"):
            st.dataframe(fi.head(100), use_container_width=True, hide_index=True)
        with st.expander("How to read support, confidence and lift"):
            st.markdown("""
- **Support** – share of respondents who use *all* items in the rule.
- **Confidence** – among respondents using the antecedent, the share who also use the consequent.
- **Lift** – confidence divided by the consequent's overall popularity. Lift > 1 means the items appear together more than chance.
- Only the 40 most common items are mined to keep the app fast. Rules show co-occurrence, not causation.""")

elif page == "Regression Analysis":
    st.subheader("Regression analysis: what is associated with developer pay?")
    reg_type = st.selectbox("Choose regression type",
                            ["Linear Regression", "Multiple Regression", "Logistic Regression"])
    data_reg = build_regression_data(filtered)
    st.caption("Target comes from ConvertedCompYearly (positive values, below the 95th percentile to limit extreme outliers). "
               "Rows with missing values in the chosen variables are dropped. Results show association, not causation.")

    if reg_type == "Linear Regression":
        st.markdown('<div class="insight"><b>Linear regression</b> fits one straight line: '
                    'Salary = intercept + slope × (one predictor).</div>', unsafe_allow_html=True)
        st.markdown("")
        x_var = st.selectbox("Predictor (X)", REG_FEATURES[:5], index=0)
        d = data_reg[[x_var, "Salary"]].dropna()
        if len(d) < 50:
            st.info("Not enough records for the current filters.")
        else:
            model = LinearRegression().fit(d[[x_var]], d["Salary"])
            pred = model.predict(d[[x_var]])
            r2 = r2_score(d["Salary"], pred)
            rmse = float(np.sqrt(mean_squared_error(d["Salary"], pred)))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Slope (USD per unit)", f"{model.coef_[0]:,.0f}")
            k2.metric("Intercept (USD)", f"{model.intercept_:,.0f}")
            k3.metric("R²", f"{r2:.3f}")
            k4.metric("RMSE (USD)", f"{rmse:,.0f}")
            samp = d.sample(min(4000, len(d)), random_state=1)
            fig = px.scatter(samp, x=x_var, y="Salary", opacity=0.35, title=f"Salary vs {x_var}")
            xs = np.linspace(d[x_var].min(), d[x_var].max(), 100)
            fig.add_trace(go.Scatter(x=xs, y=model.predict(pd.DataFrame({x_var: xs})), mode="lines",
                                     name="Regression line", line=dict(color="#ff9f1c", width=4)))
            fig.update_yaxes(title="Annual compensation (USD)")
            st.plotly_chart(style_fig(fig, height=460), use_container_width=True)
            st.markdown(
                f'<div class="insight">Each additional unit of <b>{x_var}</b> is associated with about '
                f'<b>${model.coef_[0]:,.0f}</b> more yearly pay. The line explains <b>{r2:.1%}</b> of the variation '
                f'in salary (n = {len(d):,}).</div>', unsafe_allow_html=True)

    elif reg_type == "Multiple Regression":
        st.markdown('<div class="insight"><b>Multiple regression</b> uses several predictors together: '
                    'Salary = b0 + b1·X1 + b2·X2 + … Each coefficient is the effect of that variable '
                    'holding the others constant.</div>', unsafe_allow_html=True)
        st.markdown("")
        feats = st.multiselect("Predictors (X)", REG_FEATURES,
                               default=["Years of professional coding", "Education level (1-8)",
                                        "Company size (1-9)", "Lives in USA", "Fully remote"])
        if len(feats) < 2:
            st.info("Select at least two predictors.")
        else:
            d = data_reg[feats + ["Salary"]].dropna()
            if len(d) < 100:
                st.info("Not enough complete records for the current filters.")
            else:
                Xtr, Xte, ytr, yte = train_test_split(d[feats], d["Salary"], test_size=0.2, random_state=42)
                model = LinearRegression().fit(Xtr, ytr)
                pred = model.predict(Xte)
                r2 = r2_score(yte, pred)
                adj = 1 - (1 - r2) * (len(yte) - 1) / (len(yte) - len(feats) - 1)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("R² (test set)", f"{r2:.3f}")
                k2.metric("Adjusted R²", f"{adj:.3f}")
                k3.metric("RMSE (USD)", f"{np.sqrt(mean_squared_error(yte, pred)):,.0f}")
                k4.metric("MAE (USD)", f"{mean_absolute_error(yte, pred):,.0f}")
                scaler = StandardScaler().fit(d[feats])
                std_model = LinearRegression().fit(scaler.transform(d[feats]), d["Salary"])
                coefs = pd.DataFrame({"Predictor": feats, "Coefficient (USD per unit)": model.coef_,
                                      "Standardised effect (USD per 1 SD)": std_model.coef_})
                coefs = coefs.sort_values("Standardised effect (USD per 1 SD)")
                l, r = st.columns(2)
                with l:
                    fig = px.bar(coefs, x="Standardised effect (USD per 1 SD)", y="Predictor", orientation="h",
                                 color="Standardised effect (USD per 1 SD)", color_continuous_scale="RdYlGn",
                                 color_continuous_midpoint=0, title="Relative importance of predictors")
                    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(style_fig(fig, height=430), use_container_width=True)
                with r:
                    comp = pd.DataFrame({"Actual": yte, "Predicted": pred}).sample(min(3000, len(yte)), random_state=1)
                    fig = px.scatter(comp, x="Actual", y="Predicted", opacity=0.35, title="Actual vs predicted salary (test set)")
                    mx = float(max(comp.max()))
                    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect prediction",
                                             line=dict(color="#ff9f1c", dash="dash")))
                    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(style_fig(fig, height=430), use_container_width=True)
                st.dataframe(coefs.sort_values("Standardised effect (USD per 1 SD)", ascending=False)
                             .assign(Intercept=model.intercept_).round(2), use_container_width=True, hide_index=True)
                st.caption(f"Trained on {len(Xtr):,} rows, evaluated on {len(Xte):,} held-out rows. "
                           "Predictors can be correlated (e.g. age and experience), so read individual coefficients with care.")

    else:
        st.markdown('<div class="insight"><b>Logistic regression</b> predicts a yes/no outcome. Here: '
                    'is a developer a <b>high earner</b> (salary above the chosen threshold)? '
                    'The model outputs a probability between 0 and 1.</div>', unsafe_allow_html=True)
        st.markdown("")
        c1, c2 = st.columns([2, 1])
        feats = c1.multiselect("Predictors (X)", REG_FEATURES,
                               default=["Years of professional coding", "Education level (1-8)",
                                        "Company size (1-9)", "Lives in USA", "Fully remote"], key="logit_feats")
        default_thr = float(data_reg["Salary"].median()) if not data_reg.empty else 60000.0
        thr = c2.number_input("High-earner threshold (USD / year)", min_value=1000.0, max_value=500000.0,
                              value=round(default_thr, -3), step=5000.0)
        if len(feats) < 1:
            st.info("Select at least one predictor.")
        else:
            d = data_reg[feats + ["Salary"]].dropna()
            d["HighEarner"] = (d["Salary"] > thr).astype(int)
            if len(d) < 100 or d["HighEarner"].nunique() < 2:
                st.info("Not enough records, or the threshold leaves only one class. Adjust the threshold or filters.")
            else:
                Xtr, Xte, ytr, yte = train_test_split(d[feats], d["HighEarner"], test_size=0.2,
                                                      random_state=42, stratify=d["HighEarner"])
                scaler = StandardScaler().fit(Xtr)
                model = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
                proba = model.predict_proba(scaler.transform(Xte))[:, 1]
                pred = (proba >= 0.5).astype(int)
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Accuracy", f"{accuracy_score(yte, pred):.1%}")
                k2.metric("Precision", f"{precision_score(yte, pred, zero_division=0):.1%}")
                k3.metric("Recall", f"{recall_score(yte, pred, zero_division=0):.1%}")
                k4.metric("F1 score", f"{f1_score(yte, pred, zero_division=0):.2f}")
                k5.metric("ROC AUC", f"{roc_auc_score(yte, proba):.3f}")
                l, r = st.columns(2)
                with l:
                    cm = confusion_matrix(yte, pred)
                    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                                    x=["Predicted: below", "Predicted: high"], y=["Actual: below", "Actual: high"],
                                    title="Confusion matrix (test set)")
                    fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(style_fig(fig, height=400), use_container_width=True)
                with r:
                    fpr, tpr, _ = roc_curve(yte, proba)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#4cc9f0", width=4)))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                             line=dict(color="#ff9f1c", dash="dash")))
                    fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                                      yaxis_title="True positive rate", margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(style_fig(fig, height=400), use_container_width=True)
                odds = pd.DataFrame({"Predictor": feats, "Odds ratio (per 1 SD)": np.exp(model.coef_[0])})
                odds = odds.sort_values("Odds ratio (per 1 SD)", ascending=False)
                fig = px.bar(odds.sort_values("Odds ratio (per 1 SD)"), x="Odds ratio (per 1 SD)", y="Predictor",
                             orientation="h", color="Odds ratio (per 1 SD)", color_continuous_scale="RdYlGn",
                             color_continuous_midpoint=1, title="Odds ratios (above 1 raises the chance of being a high earner)")
                fig.add_vline(x=1, line_dash="dash", line_color="#ffffff")
                fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(style_fig(fig, height=380), use_container_width=True)
                best = odds.iloc[0]
                st.markdown(
                    f'<div class="insight">{(d["HighEarner"].mean()):.0%} of the {len(d):,} respondents earn above '
                    f'${thr:,.0f}. The strongest positive predictor is <b>{best["Predictor"]}</b> '
                    f'(odds ×{best["Odds ratio (per 1 SD)"]:.2f} per standard deviation).</div>', unsafe_allow_html=True)

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
