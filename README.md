# DevScope — Developer Career & Technology Trends

An interactive Streamlit dashboard based on the Stack Overflow Developer Survey 2022.

## Project structure

```text
developer_trends_project/
├── app.py
├── eda.ipynb
├── requirements.txt
├── README.md
└── data/
    └── survey_cleaned.csv
```

## Run in VS Code

1. Extract the ZIP file.
2. Open the `developer_trends_project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment (Windows):

```bash
python -m venv .venv
.venv\Scripts\activate
```

5. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

6. Start the dashboard:

```bash
streamlit run app.py
```

7. The browser opens the local dashboard. If it does not, open the Local URL printed in the terminal (usually `http://localhost:8501`).

## Run the notebook

In VS Code, open `eda.ipynb`, choose the `.venv` Python kernel, then select **Run All**. If prompted to install notebook support, install the requirements above.

## Main dashboard sections

- Overview: response KPIs, country distribution and employment mix
- Technology Trends: current use, learning interest and interest gap
- Global Developer Map: country response ranking
- Salary & Experience: compensation patterns across professional experience bands
- Education & Employment: within-education employment percentages
- Insights & Data: missingness, preview and filtered CSV download

## Methodology notes

- The survey contains optional questions; missing values are not automatically replaced with fabricated values.
- Multi-select answers separated by semicolons are expanded before counting.
- Compensation charts use `ConvertedCompYearly`, positive values, and exclude values above the 99th percentile for chart readability.
- Experience strings such as `Less than 1 year` and `More than 50 years` are converted to approximate numeric values for grouping.
- A single 2022 survey cannot prove which technologies are *growing over time*. The app therefore shows the gap between current usage and learning interest. To measure growth, compare multiple survey years using consistent definitions.
- Country response counts are survey participation counts, not estimates of total developers in a country.

## Dataset

Stack Overflow Developer Survey 2022 public results. Refer to the original survey's licensing and attribution terms when redistributing the data.
