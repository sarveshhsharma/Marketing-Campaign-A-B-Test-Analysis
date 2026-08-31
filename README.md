# Marketing A/B Testing Analysis

End-to-end analysis of an A/B test on **588,101 users**, comparing conversion
between a real **ad** campaign and a **PSA** placeholder. The project checks
that the experiment itself was sound before testing the outcome, applies a
decision rule fixed in advance, and turns the result into business impact.
Everything is also available as an interactive Streamlit dashboard.

**Dataset:** [Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing) (Kaggle, faviovaz)

## Demo

[Video walkthrough of the dashboard](https://drive.google.com/file/d/1cwh1B9_Abw-xlTPgZy5kRKLoz8RwIjzp/view?usp=sharing)

## Key Result

| Metric | Value |
|---|---|
| Ad conversion rate | 2.55% |
| PSA conversion rate | 1.79% |
| Absolute lift | +0.77 pp |
| 95% CI on absolute lift | [0.59 pp, 0.94 pp] |
| Relative lift | 43.1% |
| 95% CI on relative lift | [30.0%, 57.5%] |
| Significance | p = 8.5e-14 (one-sided two-proportion z-test) |
| Incremental conversions | ~4,343 (95% CI: 3,316 to 5,284) |

Replacing the PSA with the ad increased conversion by 0.77 pp, a 43% relative
lift. The result clears both the statistical bar and the practical threshold
that were set before the analysis, so the recommendation is to ship.

**What is actually being measured.** The control group saw a public service
announcement, not a blank page. So this estimates the value of the ad creative
over a placebo ad, not the value of advertising against no advertising at all.

## Decision rule (fixed before the analysis)

Writing the rule down first stops the analysis from drifting towards whatever
result the data happens to support.

- **Primary metric:** conversion rate, one row per user
- **Hypothesis:** one-sided, `H1: p_ad > p_psa`
- **Significance level:** alpha = 0.05
- **Practical threshold:** 0.25 pp, about a 14% relative lift on the 1.79% baseline
- **Ship only if both bars clear:** p below alpha **and** the lower CI bound above 0.25 pp

Checking the lower bound rather than the point estimate stops a noisy win from
passing.

## Workflow

1. **Design and decision rule** — pre-registered hypothesis, alpha, and practical threshold
2. **Sample Ratio Mismatch check** — does the observed split match the intended one
3. **Power analysis** — what size of effect this design could actually detect
4. **Inspection and sanity checks** — schema, duplicate users, group sizes, nulls
5. **Exploratory data analysis** — ad exposure distribution, conversion by day and hour
6. **Post-treatment diagnostics** — exposure comparison with effect sizes, clearly labelled as diagnostics rather than balance checks
7. **Significance testing** — one-sided z-test, Newcombe CI, separate CI for the relative lift, chi-square cross-check
8. **Business impact** — incremental conversions, revenue sensitivity, and per-day effects with intervals and a multiple-comparison correction

## Methodology notes

- **Sample Ratio Mismatch is checked first.** The split is a deliberate 96/4, so the SRM test compares against 96/4 rather than 50/50. It passes at `p = 0.9998`. If this check had failed, nothing downstream would be trustworthy regardless of the p-value on conversion.

- **The 96/4 split is expensive.** Only 23,524 users landed in control and just 420 converted, and the smaller group sets the precision of the whole experiment. At 80% power this design detects 0.219 pp. An even split of the same traffic would have detected 0.086 pp, roughly 2.6x smaller, for free. The observed effect clears the MDE comfortably, so it did not cost us the answer here, but it is the first thing to fix next time.

- **Exposure variables are post-treatment, not covariates.** `total ads`, `most ads day` and `most ads hour` are all recorded while the campaign runs, so the treatment can change them. A variable measured after assignment cannot validate that assignment. They are reported as descriptive diagnostics, and the randomisation evidence is the SRM check. The dataset has no pre-treatment fields at all, which also rules out a covariate-adjusted estimate such as CUPED.

- **Newcombe intervals instead of Wald.** Both conversion rates are under 3% and control has only 420 conversions, and in that corner the Wald interval undercovers.

- **The relative lift gets its own interval.** A ratio is not a linear function of a difference, so the interval on the absolute lift cannot be rescaled into one for the relative lift. Computed on the log ratio scale it comes out at [30.0%, 57.5%], which is wide, and the width traces straight back to the small control group.

- **Effect sizes alongside p-values.** With 588K rows almost any difference becomes significant, so Cramér's V and raw medians are reported next to every chi-square result.

- **Mann-Whitney U rather than a t-test** for ad exposure, since `total ads` is heavily right-skewed (median 13, mean 24.8, max 2,065).

- **Holm correction on the per-day breakdown.** Seven days means seven tests and seven chances to get lucky. After correction 5 of 7 days still show a significant lift; Thursday and Sunday do not, and their intervals cross zero. Friday moves from `p = 0.0116` raw to `0.0347` corrected, which shows how easily an uncorrected segment scan manufactures findings.

- **Revenue as a sensitivity table.** The dataset has no price or margin column, so revenue is shown across a range of assumed margins instead of one invented number.

## Project structure

```
.
├── app.py                    # Streamlit dashboard, 6 tabs following the workflow
├── src/
│   └── analysis.py           # All statistical logic, shared by the app and the notebook
├── data/
│   └── marketing_AB.csv      # Raw dataset, not committed
├── notebooks/
│   └── AB.ipynb              # Full analysis and write-up
└── requirements.txt
```

The tests live in `src/analysis.py` so the notebook and the dashboard cannot
drift apart and report different numbers for the same thing.

## Tech stack

Python, pandas, numpy, scipy, statsmodels, matplotlib, Streamlit

## Running locally

Download the dataset from Kaggle and save it as `data/marketing_AB.csv`, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- No pre-treatment covariates exist in the data, so randomisation can only be checked through the SRM test and no covariate-adjusted estimate is possible
- Day-of-week effects are tested per segment with a Holm correction, but no formal day-by-treatment interaction test is run
- The data has only day of week and hour, no calendar dates, so novelty effects and time trends across the campaign cannot be checked
- The control is a PSA rather than no ad, so the estimate is the value of the creative over a placebo, not the value of advertising itself
