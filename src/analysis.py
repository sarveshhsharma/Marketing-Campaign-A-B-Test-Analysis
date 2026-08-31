"""Shared stats helpers for the marketing A/B test.

Both the notebook and the Streamlit app import from here so the numbers
reported in each place always come from the same code.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)

# The experiment was designed with a 96/4 split, not 50/50.
EXPECTED_AD_SHARE = 0.96

# Decision rule fixed before looking at the conversion outcome.
ALPHA = 0.05
PRACTICAL_THRESHOLD_PP = 0.25

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def load_data(path="data/marketing_AB.csv"):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def get_counts(df):
    """Return conversions and users for the ad and psa groups."""
    counts = df.groupby("test group")["converted"].agg(["sum", "count"])
    return {
        "x_ad": int(counts.loc["ad", "sum"]),
        "n_ad": int(counts.loc["ad", "count"]),
        "x_psa": int(counts.loc["psa", "sum"]),
        "n_psa": int(counts.loc["psa", "count"]),
    }


def srm_test(n_ad, n_psa, expected_ad_share=EXPECTED_AD_SHARE):
    """Sample ratio mismatch check.

    Compares the observed split against the split the experiment was
    supposed to use. A small p-value means users were not assigned in the
    intended proportion, which usually points to a bug in the assignment or
    logging rather than to a real treatment effect.
    """
    observed = np.array([n_ad, n_psa])
    expected = np.array([expected_ad_share, 1 - expected_ad_share]) * observed.sum()
    chi2, p_value = stats.chisquare(observed, expected)
    return {
        "observed_ad_share": n_ad / observed.sum(),
        "expected_ad_share": expected_ad_share,
        "chi2": chi2,
        "p_value": p_value,
        "passed": p_value > 0.001,
    }


def two_proportion_test(x_ad, n_ad, x_psa, n_psa, alpha=ALPHA):
    """One-sided z-test for ad > psa, with a Newcombe CI on the difference.

    Newcombe is used instead of Wald because conversion rates here are under
    3 percent and the control group has only a few hundred conversions, where
    the Wald interval is known to undercover.
    """
    p_ad = x_ad / n_ad
    p_psa = x_psa / n_psa

    z_stat, p_one_sided = proportions_ztest(
        [x_ad, x_psa], [n_ad, n_psa], alternative="larger"
    )
    _, p_two_sided = proportions_ztest(
        [x_ad, x_psa], [n_ad, n_psa], alternative="two-sided"
    )

    ci_low, ci_high = confint_proportions_2indep(
        x_ad, n_ad, x_psa, n_psa, compare="diff", method="newcomb", alpha=alpha
    )

    return {
        "p_ad": p_ad,
        "p_psa": p_psa,
        "diff": p_ad - p_psa,
        "z_stat": z_stat,
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z_critical": stats.norm.ppf(1 - alpha),
    }


def relative_lift_ci(x_ad, n_ad, x_psa, n_psa, alpha=ALPHA):
    """CI for the relative lift, built from the ratio of the two rates.

    The headline "43 percent lift" is a ratio, so it needs its own interval.
    Transforming the absolute CI would give the wrong answer because the
    ratio is not a linear function of the difference.
    """
    ratio_low, ratio_high = confint_proportions_2indep(
        x_ad, n_ad, x_psa, n_psa, compare="ratio", method="log", alpha=alpha
    )
    p_ad = x_ad / n_ad
    p_psa = x_psa / n_psa
    return {
        "ratio": p_ad / p_psa,
        "lift": p_ad / p_psa - 1,
        "lift_low": ratio_low - 1,
        "lift_high": ratio_high - 1,
    }


def minimum_detectable_effect(baseline_rate, n_ad, n_psa, alpha=ALPHA, power=0.8):
    """Smallest absolute lift this design could reliably pick up.

    Uses the one-sided alpha to match the primary test. The unequal 96/4
    split matters a lot here because the standard error is dominated by the
    much smaller control group.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_power = stats.norm.ppf(power)
    se = np.sqrt(baseline_rate * (1 - baseline_rate) * (1 / n_ad + 1 / n_psa))
    mde_abs = (z_alpha + z_power) * se

    # What an even split of the same total traffic would have bought us.
    n_even = (n_ad + n_psa) / 2
    se_even = np.sqrt(baseline_rate * (1 - baseline_rate) * (2 / n_even))
    mde_even = (z_alpha + z_power) * se_even

    return {
        "mde_abs": mde_abs,
        "mde_relative": mde_abs / baseline_rate,
        "mde_abs_balanced": mde_even,
        "power": power,
    }


def cramers_v(contingency_table):
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.values.sum()
    rows, columns = contingency_table.shape
    return np.sqrt(chi2 / (n * min(rows - 1, columns - 1)))


def post_treatment_diagnostics(df):
    """Compare exposure variables across groups.

    These are not randomisation checks. total ads, most ads day and most ads
    hour are all recorded while the campaign is running, so the treatment can
    change them. They are reported as descriptive diagnostics only.
    """
    ad_total = df[df["test group"] == "ad"]["total ads"]
    psa_total = df[df["test group"] == "psa"]["total ads"]

    u_stat, p_mw = stats.mannwhitneyu(ad_total, psa_total, alternative="two-sided")

    day_ct = pd.crosstab(df["test group"], df["most ads day"])
    hour_ct = pd.crosstab(df["test group"], df["most ads hour"])

    return {
        "ad_median": ad_total.median(),
        "psa_median": psa_total.median(),
        "ad_mean": ad_total.mean(),
        "psa_mean": psa_total.mean(),
        "mannwhitney_u": u_stat,
        "mannwhitney_p": p_mw,
        "cramers_v_day": cramers_v(day_ct),
        "cramers_v_hour": cramers_v(hour_ct),
    }


def segment_lifts(df, segment_col="most ads day", order=None, alpha=ALPHA):
    """Per-segment lift with confidence intervals and a Holm correction.

    Running one test per day of the week means seven chances to find something
    by luck, so the raw p-values are adjusted before anything is called real.
    """
    if order is None:
        order = DAY_ORDER

    rows = []
    for segment in order:
        part = df[df[segment_col] == segment]
        counts = part.groupby("test group")["converted"].agg(["sum", "count"])
        x_ad = int(counts.loc["ad", "sum"])
        n_ad = int(counts.loc["ad", "count"])
        x_psa = int(counts.loc["psa", "sum"])
        n_psa = int(counts.loc["psa", "count"])

        p_ad = x_ad / n_ad
        p_psa = x_psa / n_psa
        _, p_value = proportions_ztest(
            [x_ad, x_psa], [n_ad, n_psa], alternative="two-sided"
        )
        ci_low, ci_high = confint_proportions_2indep(
            x_ad, n_ad, x_psa, n_psa, compare="diff", method="newcomb", alpha=alpha
        )

        rows.append(
            {
                "segment": segment,
                "ad_rate": p_ad,
                "psa_rate": p_psa,
                "psa_users": n_psa,
                "lift_pp": (p_ad - p_psa) * 100,
                "ci_low_pp": ci_low * 100,
                "ci_high_pp": ci_high * 100,
                "p_raw": p_value,
            }
        )

    table = pd.DataFrame(rows)
    reject, p_adj = multipletests(table["p_raw"], alpha=alpha, method="holm")[:2]
    table["p_holm"] = p_adj
    table["significant"] = reject
    return table


def decision(test_result, threshold_pp=PRACTICAL_THRESHOLD_PP, alpha=ALPHA):
    """Apply the pre-registered rule to the primary test result.

    Ship only if the result clears both bars: statistically significant and a
    lower confidence bound above the practical threshold. Checking the lower
    bound rather than the point estimate keeps a noisy win from passing.
    """
    stat_pass = test_result["p_one_sided"] < alpha
    practical_pass = test_result["ci_low"] * 100 > threshold_pp
    return {
        "stat_pass": stat_pass,
        "practical_pass": practical_pass,
        "ship": stat_pass and practical_pass,
        "threshold_pp": threshold_pp,
    }


def business_impact(test_result, n_ad, margins=(20, 50, 100, 200)):
    """Incremental conversions and revenue across assumed margins.

    The dataset has no price or margin column, so revenue is shown as a
    sensitivity table instead of a single made up number.
    """
    point = test_result["diff"] * n_ad
    low = test_result["ci_low"] * n_ad
    high = test_result["ci_high"] * n_ad

    revenue = pd.DataFrame(
        [
            {
                "Margin per conversion": f"${m}",
                "Point estimate": point * m,
                "CI low": low * m,
                "CI high": high * m,
            }
            for m in margins
        ]
    )

    return {
        "incremental_point": point,
        "incremental_low": low,
        "incremental_high": high,
        "revenue": revenue,
    }
