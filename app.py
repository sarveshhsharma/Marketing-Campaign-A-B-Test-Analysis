import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import analysis as an

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "marketing_AB.csv"
)

st.set_page_config(page_title="Marketing A/B Testing", layout="wide")
st.title("Marketing A/B Testing Analysis")
st.caption(
    "Ad campaign vs PSA placeholder, 588,101 users. The control group saw a "
    "public service announcement, not a blank page, so the effect below is the "
    "value of the ad creative over a placebo ad."
)


@st.cache_data
def get_data():
    return an.load_data(DATA_PATH)


@st.cache_data
def get_results():
    """Run every test once and reuse the output across all tabs."""
    df = get_data()
    counts = an.get_counts(df)
    test = an.two_proportion_test(**counts)
    return {
        "counts": counts,
        "test": test,
        "srm": an.srm_test(counts["n_ad"], counts["n_psa"]),
        "relative": an.relative_lift_ci(**counts),
        "mde": an.minimum_detectable_effect(
            test["p_psa"], counts["n_ad"], counts["n_psa"]
        ),
        "decision": an.decision(test),
        "diagnostics": an.post_treatment_diagnostics(df),
        "segments": an.segment_lifts(df),
        "impact": an.business_impact(test, counts["n_ad"]),
    }


try:
    df = get_data()
    res = get_results()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'data/marketing_AB.csv' exists.")
    st.stop()

counts = res["counts"]
test = res["test"]
rel = res["relative"]
decision = res["decision"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Design & Decision Rule",
        "Inspection & Sanity Checks",
        "EDA",
        "Post-Treatment Diagnostics",
        "Significance Testing",
        "Business Impact",
    ]
)

# ==========================================
# TAB 1: DESIGN AND DECISION RULE
# ==========================================
with tab1:
    st.header("Experiment Design and Decision Rule")

    st.subheader("1. Decision rule, fixed before looking at the outcome")
    st.markdown(
        f"""
Writing the rule down first stops the analysis from drifting towards whatever
result the data happens to support.

- **Primary metric:** conversion rate, one row per user.
- **Hypothesis:** one-sided, `H1: p_ad > p_psa`. The business question is
  whether the ad beats the PSA, and a worse ad gets shelved either way.
- **Significance level:** alpha = {an.ALPHA}.
- **Practical threshold:** the campaign is only worth running if the true lift
  is at least **{an.PRACTICAL_THRESHOLD_PP} pp**. On a {test['p_psa']:.2%} baseline that is about a
  {an.PRACTICAL_THRESHOLD_PP / (test['p_psa'] * 100):.0%} relative lift, set to cover the operational cost of
  serving the campaign.
- **Ship only if both bars clear:** p-value below alpha **and** the lower bound
  of the 95% CI above the practical threshold. Using the lower bound instead of
  the point estimate stops a noisy win from passing.
"""
    )

    srm = res["srm"]

    st.subheader("2. Sample Ratio Mismatch check")
    col1, col2, col3 = st.columns(3)
    col1.metric("Observed ad share", f"{srm['observed_ad_share']:.2%}")
    col2.metric("Intended ad share", f"{srm['expected_ad_share']:.0%}")
    col3.metric("SRM p-value", f"{srm['p_value']:.4f}")

    if srm["passed"]:
        st.success(
            "No sample ratio mismatch. The observed split matches the intended "
            "96/4 allocation, so there is no sign of a broken assignment or "
            "logging pipeline. This is the first thing to check, because an SRM "
            "makes everything downstream untrustworthy no matter how small the "
            "p-value on the primary metric turns out to be."
        )
    else:
        st.error(
            "Sample ratio mismatch detected. Investigate assignment before "
            "reading any result below."
        )

    st.subheader("3. The 96/4 split costs real power")
    mde = res["mde"]
    col4, col5, col6 = st.columns(3)
    col4.metric("Control users", f"{counts['n_psa']:,}")
    col5.metric("Control conversions", f"{counts['x_psa']:,}")
    col6.metric("MDE at 80% power", f"{mde['mde_abs'] * 100:.3f} pp")

    st.markdown(
        f"""
Only **{counts['n_psa']:,} users** landed in the control group, and just **{counts['x_psa']:,} of them
converted**. The standard error of the difference is driven by whichever group
is smaller, so that control group sets the precision of the whole experiment.

At 80% power and a one-sided alpha of {an.ALPHA}, this design detects an absolute lift
of **{mde['mde_abs'] * 100:.3f} pp** ({mde['mde_relative']:.1%} relative). Splitting the same
{counts['n_ad'] + counts['n_psa']:,} users evenly would have detected **{mde['mde_abs_balanced'] * 100:.3f} pp**, about
**{mde['mde_abs'] / mde['mde_abs_balanced']:.1f}x smaller**, at no extra traffic cost.

The observed effect clears the MDE comfortably, so the imbalance did not cost us
the answer this time. It would have mattered for a smaller effect, and it is the
first thing to fix in the next test.
"""
    )

    st.subheader("4. Verdict against the pre-registered rule")
    col7, col8 = st.columns(2)
    col7.metric(
        "Statistical bar",
        "Pass" if decision["stat_pass"] else "Fail",
        f"p = {test['p_one_sided']:.2e}",
    )
    col8.metric(
        "Practical bar",
        "Pass" if decision["practical_pass"] else "Fail",
        f"CI low {test['ci_low'] * 100:.3f} pp vs {decision['threshold_pp']} pp",
    )
    if decision["ship"]:
        st.success("Both bars cleared. Recommendation: ship the ad campaign.")
    else:
        st.warning("The pre-registered rule was not met. Do not ship on this evidence.")

# ==========================================
# TAB 2: INSPECTION & SANITY CHECKS
# ==========================================
with tab2:
    st.header("Data Inspection & Sanity Checks")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Unique Users", f"{df['user id'].nunique():,}")
    col3.metric("Duplicate Users", f"{df['user id'].duplicated().sum()}")

    st.subheader("Group Sizes")
    group_counts = df["test group"].value_counts()
    group_pct = (group_counts / len(df) * 100).round(2)
    st.dataframe(pd.DataFrame({"Count": group_counts, "Percentage (%)": group_pct}))

    st.subheader("Overall Conversions")
    col4, col5 = st.columns(2)
    col4.metric("Total Converted Users", f"{df['converted'].sum():,}")
    col5.metric("Overall Conversion Rate", f"{df['converted'].mean():.2%}")

    st.write("**Group Summary:**")
    group_summary = df.groupby("test group")["converted"].agg(
        users="count", conversions="sum", conversion_rate="mean"
    )
    group_summary["conversion_rate"] *= 100
    st.dataframe(group_summary.style.format({"conversion_rate": "{:.2f}%"}))

    st.info(
        "One row per user id and no duplicates, so each user contributes a single "
        "independent observation. That is what lets the two-proportion z-test be "
        "used directly with no clustering correction."
    )

# ==========================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab3:
    st.header("Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Ads - Overall")
        st.dataframe(df["total ads"].describe())
    with col2:
        st.subheader("Total Ads - By Group")
        st.dataframe(df.groupby("test group")["total ads"].describe())

    st.subheader("Visualizations")
    col3, col4 = st.columns(2)

    with col3:
        conversion_by_day = (
            df.groupby("most ads day")["converted"].mean().reindex(an.DAY_ORDER) * 100
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(conversion_by_day.index, conversion_by_day.values)
        ax.bar_label(
            bars, labels=[f"{v:.2f}%" for v in conversion_by_day.values], padding=3
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("Conversion Rate by Day")
        ax.set_ylim(0, conversion_by_day.max() + 0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    with col4:
        conversion_by_hour = df.groupby("most ads hour")["converted"].mean() * 100
        x = conversion_by_hour.index.values
        y = conversion_by_hour.values

        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = make_interp_spline(x, y, k=3)(x_smooth)

        fig2 = plt.figure(figsize=(12, 5))
        plt.plot(x_smooth, y_smooth, linewidth=2)
        plt.fill_between(x_smooth, y_smooth, alpha=0.2)
        plt.xlabel("Hour")
        plt.ylabel("Conversion Rate (%)")
        plt.title("Conversion Rate by Hour")
        plt.xticks(range(0, 24))
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown(
        """
### Exploratory Analysis

- **Right-skewed ad exposure:** `total ads` has a median of 13 against a mean of
  about 24.8 and a maximum of 2,065. A small group of users saw a very large
  number of ads, which is why a plain t-test is not used on that column later.
- **Conversion rate by day:** conversion peaks on Monday and Tuesday near 3% and
  dips to about 2.1% on Thursday and Saturday.
- **Conversion rate by hour:** lowest overnight at roughly 0.7% around 2 AM, then
  climbing to about 3% between 4 PM and 8 PM.

> **Important:** these curves pool both groups, so they describe when people
> convert in general, not when ads work better. The per-day treatment effect is
> tested properly in the Business Impact tab.
"""
    )

# ==========================================
# TAB 4: POST-TREATMENT DIAGNOSTICS
# ==========================================
with tab4:
    st.header("Post-Treatment Diagnostics")

    st.warning(
        "These are not randomisation checks. `total ads`, `most ads day` and "
        "`most ads hour` are all recorded while the campaign runs, so the "
        "treatment itself can change them. A variable measured after assignment "
        "cannot validate that assignment, and conditioning on one can introduce "
        "bias rather than remove it. The real randomisation evidence is the SRM "
        "check in the first tab."
    )

    diag = res["diagnostics"]

    st.subheader("1. Ad exposure between groups")
    col1, col2 = st.columns(2)
    with col1:
        st.write(
            f"**Median total ads (Ad / PSA):** {diag['ad_median']:.1f} / {diag['psa_median']:.1f}"
        )
        st.write(
            f"**Mean total ads (Ad / PSA):** {diag['ad_mean']:.2f} / {diag['psa_mean']:.2f}"
        )
        st.write(f"**Mann-Whitney U p-value:** `{diag['mannwhitney_p']:.2e}`")
    with col2:
        st.write(f"**Cramér's V (Day vs Group):** {diag['cramers_v_day']:.4f}")
        st.write(f"**Cramér's V (Hour vs Group):** {diag['cramers_v_hour']:.4f}")

    st.markdown(
        f"""
### Interpretation

Mann-Whitney is used instead of a t-test because `total ads` is heavily skewed,
and it returns `p = {diag['mannwhitney_p']:.2e}`. That looks alarming until you check how big
the gap actually is. The medians are {diag['ad_median']:.0f} and {diag['psa_median']:.0f}, and the means differ by
about {abs(diag['ad_mean'] - diag['psa_mean']):.2f} ads. With 588K rows a difference that small is still
"significant", which is exactly why effect sizes sit next to the p-values here.
Cramér's V is {diag['cramers_v_day']:.4f} for day and {diag['cramers_v_hour']:.4f} for hour, both negligible.

**Conclusion:** exposure patterns look practically identical across groups. That
is reassuring, but it describes what happened during the experiment rather than
proving assignment was random. Read it alongside the SRM result, not instead
of it.

**What is missing:** the dataset has no pre-treatment fields at all, no age,
tenure, past purchases or acquisition channel. A proper balance table cannot be
built from this data, and a covariate-adjusted estimate such as CUPED is off the
table for the same reason. That is a limitation worth naming rather than
papering over.
"""
    )

# ==========================================
# TAB 5: SIGNIFICANCE TESTING
# ==========================================
with tab5:
    st.header("Significance Testing")

    col1, col2, col3 = st.columns(3)
    col1.metric("Ad Conversion Rate", f"{test['p_ad']:.4%}")
    col2.metric("PSA Conversion Rate", f"{test['p_psa']:.4%}")
    col3.metric("Absolute Lift", f"{test['diff'] * 100:.3f} pp")

    st.subheader("Statistical Results")
    st.write(f"- **Observed Z-statistic:** `{test['z_stat']:.4f}`")
    st.write(f"- **One-sided p-value:** `{test['p_one_sided']:.2e}`")
    st.write(f"- **Absolute difference (Ad - PSA):** `{test['diff'] * 100:.3f} pp`")
    st.write(
        f"- **95% CI on the difference:** "
        f"`[{test['ci_low'] * 100:.3f} pp, {test['ci_high'] * 100:.3f} pp]`"
    )
    st.write(
        f"- **Relative lift:** `{rel['lift']:.1%}` "
        f"(95% CI `[{rel['lift_low']:.1%}, {rel['lift_high']:.1%}]`)"
    )

    st.info(
        "The interval on the difference uses the Newcombe method rather than "
        "Wald. Both rates are under 3% and the control group has only 420 "
        "conversions, and in that corner the Wald interval is known to fall "
        "short of its stated 95% coverage."
    )

    st.subheader("Why the relative lift needs its own interval")
    st.markdown(
        f"""
The number people repeat is the **{rel['lift']:.0%} relative lift**, but a ratio is not a
linear function of a difference, so the interval on the difference cannot simply
be rescaled. Computing it on the log ratio scale gives
**[{rel['lift_low']:.1%}, {rel['lift_high']:.1%}]**.

That interval is wide. The honest summary is not "ads lift conversion by {rel['lift']:.0%}",
it is "somewhere between a {rel['lift_low']:.0%} and a {rel['lift_high']:.0%} lift, most likely near {rel['lift']:.0%}". The
width comes from the small control group, which brings the story back to the
96/4 split.
"""
    )

    z_stat = test["z_stat"]
    z_critical = test["z_critical"]
    z = np.linspace(-4, max(5, z_stat + 1), 3000)

    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(z, stats.norm.pdf(z), linewidth=2, label="Null distribution")
    z_pvalue = z[z >= z_stat]
    plt.fill_between(
        z_pvalue,
        stats.norm.pdf(z_pvalue),
        0,
        alpha=0.5,
        label=f"p-value region (p = {test['p_one_sided']:.2e})",
    )
    z_rejection = z[z >= z_critical]
    plt.fill_between(
        z_rejection,
        stats.norm.pdf(z_rejection),
        0,
        alpha=0.15,
        label=f"Rejection region (alpha = {an.ALPHA})",
    )
    plt.axvline(z_stat, linestyle="--", linewidth=2, label=f"Observed Z = {z_stat:.2f}")
    plt.axvline(
        z_critical, linestyle=":", linewidth=2, label=f"Critical Z = {z_critical:.2f}"
    )
    plt.xlabel("Z-statistic")
    plt.ylabel("Probability Density")
    plt.title("Two-Proportion Z-Test: Null Distribution\nAd Group vs PSA Group")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown(
        f"""
### Verdict: Statistically Significant

The p-value sits far below alpha, so the null of equal conversion rates is
rejected. The effect also clears the {an.PRACTICAL_THRESHOLD_PP} pp practical threshold fixed in advance,
since the lower bound of the CI is {test['ci_low'] * 100:.3f} pp.

With 588K rows a tiny difference would have been "significant" too, which is why
the decision rule was written around the confidence interval rather than the
p-value alone.
"""
    )

# ==========================================
# TAB 6: BUSINESS IMPACT
# ==========================================
with tab6:
    st.header("Business Impact")

    impact = res["impact"]
    n_ad = counts["n_ad"]

    st.markdown(
        f"""
### 1. Incremental Conversions
Across the {n_ad:,} users in the `ad` group, the treatment generated:
* **Point estimate:** ~{impact['incremental_point']:,.0f} conversions
* **95% CI:** {impact['incremental_low']:,.0f} to {impact['incremental_high']:,.0f} additional conversions
"""
    )

    st.subheader("2. Revenue Sensitivity Analysis")
    st.caption(
        "The dataset has no price or margin column, so revenue is shown across a "
        "range of assumed margins instead of one invented number."
    )
    revenue = impact["revenue"].copy()
    for col in ["Point estimate", "CI low", "CI high"]:
        revenue[col] = revenue[col].map(lambda v: f"${v:,.0f}")
    st.table(revenue)

    st.subheader("3. Does the lift hold on every day of the week?")
    segments = res["segments"]

    display = pd.DataFrame(
        {
            "Day": segments["segment"],
            "Ad rate": segments["ad_rate"].map("{:.2%}".format),
            "PSA rate": segments["psa_rate"].map("{:.2%}".format),
            "Control users": segments["psa_users"].map("{:,}".format),
            "Lift": segments["lift_pp"].map("{:+.2f} pp".format),
            "95% CI": [
                f"[{lo:+.2f}, {hi:+.2f}]"
                for lo, hi in zip(segments["ci_low_pp"], segments["ci_high_pp"])
            ],
            "p (Holm)": segments["p_holm"].map("{:.4f}".format),
            "Holds up": np.where(segments["significant"], "Yes", "No"),
        }
    )
    st.table(display)

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    # errorbar takes one colour per call, so each day is drawn separately.
    for i, row in segments.iterrows():
        color = "tab:blue" if row["significant"] else "tab:gray"
        ax4.errorbar(
            row["lift_pp"],
            i,
            xerr=[[row["lift_pp"] - row["ci_low_pp"]], [row["ci_high_pp"] - row["lift_pp"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
        )
    ax4.axvline(0, linestyle="--", linewidth=1, color="black")
    ax4.set_yticks(np.arange(len(segments)))
    ax4.set_yticklabels(segments["segment"])
    ax4.invert_yaxis()
    ax4.set_xlabel("Lift in conversion rate (pp)")
    ax4.set_title(
        "Per-day lift with 95% CI\nGrey means not significant after Holm correction"
    )
    ax4.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    st.pyplot(fig4)

    n_sig = int(segments["significant"].sum())
    not_sig = " and ".join(segments.loc[~segments["significant"], "segment"])
    st.markdown(
        f"""
Splitting by day means running seven tests, so seven chances to find something
by luck. The p-values above are corrected with the Holm method, which controls
the family-wise error rate and is uniformly more powerful than Bonferroni.

**{n_sig} of 7 days** still show a significant lift after correction. **{not_sig}**
do not, and their intervals cross zero. Each day has only around 3,000 control
users, so the per-day intervals are wide, and this is mostly a statement about
sample size rather than evidence that ads stop working midweek.

Treat this as a hypothesis for the next experiment, not as a scheduling
decision. A formal day-by-treatment interaction test would be the right way to
claim the effect genuinely varies by day, and that is not run here.
"""
    )

    st.subheader("4. Stakeholder Summary")
    st.markdown(
        f"""
> Replacing the PSA with the ad increased conversion by **{test['diff'] * 100:.2f} pp**
> (95% CI **{test['ci_low'] * 100:.2f} to {test['ci_high'] * 100:.2f} pp**), a relative lift of
> **{rel['lift']:.0%}** (95% CI **{rel['lift_low']:.0%} to {rel['lift_high']:.0%}**), with `p < 0.001`. That is
> roughly **{impact['incremental_point']:,.0f} extra conversions** across {n_ad:,} users. The
> result clears both the statistical bar and the {decision['threshold_pp']} pp practical bar set
> before the analysis, so the recommendation is to ship.
>
> Two caveats worth carrying into the next test. The control group was only 4%
> of traffic, which is why the relative lift interval is wide. And the control
> saw a PSA rather than nothing, so this measures the ad against a placebo ad,
> not against no advertising at all.
"""
    )
