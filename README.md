# Marketing A/B Testing Analysis

## Demo
[View Demo](https://drive.google.com/file/d/1cwh1B9_Abw-xlTPgZy5kRKLoz8RwIjzp/view?usp=sharing)

End-to-end A/B test analysis on **588,101 users**, comparing conversion rates between a real **ad** campaign (treatment) and a **PSA** placeholder (control). The project validates experiment integrity before testing, runs a rigorous significance test, and translates the result into business impact — deployed as an interactive Streamlit dashboard.

**Dataset:** [Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) (Kaggle, faviovaz)

## Key Result

| Metric | Value |
|---|---|
| Ad conversion rate | 2.55% |
| PSA conversion rate | 1.79% |
| Absolute lift | +0.77 percentage points |
| Relative lift | ~43% |
| 95% CI on difference | [0.60pp, 0.94pp] |
| Significance | p < 0.001 (two-proportion z-test) |

Replacing the PSA with the ad increased conversion by **0.77pp** (95% CI: 0.60–0.94pp), a **~43% relative lift**, and the result is statistically significant.

## Workflow

1. **Inspection & Sanity Checks** — schema validation, duplicate-user check, group sizes, null checks
2. **Exploratory Data Analysis** — ad exposure distribution, conversion patterns by day/hour
3. **Randomization & Covariate Balance** — Mann-Whitney U and Chi-square tests to confirm groups aren't confounded, backed by effect sizes (Cramér's V) rather than p-values alone
4. **Significance Testing** — two-proportion z-test with a 95% confidence interval, cross-checked against a chi-square test
5. **Business Impact** — incremental conversions, a revenue sensitivity table across assumed margins, and treatment-effect variation by day of week

## Project Structure

```
.
├── app.py                    # Streamlit dashboard (5 tabs, mirrors the workflow above)
├── data/
│   └── marketing_AB.csv      # Raw dataset
├── notebooks/
│   └── AB.ipynb              # Full exploratory analysis and write-up
└── requirements.txt
```

## Tech Stack

Python · pandas · numpy · scipy · statsmodels · matplotlib · seaborn · Streamlit

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Methodology Notes

- **Mann-Whitney U** (not a t-test) was used to compare ad exposure across groups, since `total ads` is heavily right-skewed (median 13, mean 24.8, max 2,065).
- **Cramér's V** was used alongside chi-square p-values for covariate balance checks — with ~588K rows, even trivial differences become "significant," so effect size was needed to judge practical relevance.
- A **one-sided z-test** was used for the primary hypothesis (ad conversion > PSA conversion), consistent with the directional business question, and cross-validated against a two-sided chi-square test.
- Revenue impact is presented as a **sensitivity table** across assumed margins per conversion, since the dataset has no price/margin field.

## Limitations

- No correction for multiple comparisons across the covariate balance tests
- Day-of-week lift differences are descriptive, not formally tested for interaction significance
- No covariate-adjusted (e.g. logistic regression) estimate of the treatment effect
