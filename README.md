# DevScope — Developer Career & Technology Trends

An interactive Streamlit dashboard based on the Stack Overflow Developer Survey 2022.

## Project structure

```text
DevScope_Streamlit_Ready/
├── app.py
├── eda.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── survey_cleaned.csv.gz
```

**Important:** Keep `survey_cleaned.csv.gz` inside the `data` folder. Do not rename it or upload it beside `app.py`.
The app reads the compressed CSV directly with Pandas; you do not need to extract it.

## Run locally

1. Extract this project ZIP.
2. Open the `DevScope_Streamlit_Ready` folder in VS Code.
3. Open Terminal and run:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Extract this ZIP on your computer.
2. Create a GitHub repository.
3. Upload the **contents** of `DevScope_Streamlit_Ready` so that `app.py`, `requirements.txt`, and the `data` folder are visible at the repository's top level.
4. Confirm that this exact file exists in GitHub: `data/survey_cleaned.csv.gz`.
5. On Streamlit Community Cloud, choose the repository and set the app file path to `app.py`.
6. Deploy.

Do not upload only `app.py`, and do not leave the dataset inside the original `survey_cleaned.zip`. GitHub stores the compressed dataset; the app reads it directly.

## Main dashboard sections

- Overview: response KPIs, country distribution and employment mix
- Technology Trends: current use, learning interest and interest gap
- Global Developer Map: country response ranking
- Salary & Experience: compensation patterns across professional experience bands
- Education & Employment: within-education employment percentages
- Apriori Association Rules: frequent itemsets and rules (support, confidence, lift) on technologies selected together
- Regression Analysis: dropdown for Linear, Multiple and Logistic regression with metrics and charts
- Insights & Data: missingness, preview and filtered CSV download

## Methodology notes

- The survey contains optional questions; missing values are not automatically replaced with fabricated values.
- Multi-select answers separated by semicolons are expanded before counting.
- Compensation charts use `ConvertedCompYearly`, positive values, and exclude values above the 99th percentile for chart readability.
- Experience strings such as `Less than 1 year` and `More than 50 years` are converted to approximate numeric values for grouping.
- A single 2022 survey cannot prove which technologies are growing over time. The app shows the gap between current usage and learning interest.
- Country response counts are survey participation counts, not estimates of total developers in a country.

- Apriori is implemented from scratch (join + prune steps) on the 40 most common items; each respondent is one transaction.
- Regression uses ConvertedCompYearly below the 95th percentile; logistic regression predicts salary above a chosen threshold (default: median).

## Dataset

Stack Overflow Developer Survey 2022 public results. Refer to the original survey's licensing and attribution terms when redistributing the data.
