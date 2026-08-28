import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from scipy.stats import chi2_contingency
from scipy.interpolate import make_interp_spline
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Marketing A/B Testing", layout="wide")
st.title("Marketing A/B Testing Analysis")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/marketing_AB.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'data/marketing_AB.csv' exists.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Inspection & Sanity Checks", 
    "EDA", 
    "Covariate Balance", 
    "Significance Testing", 
    "Business Impact"
])

# ==========================================
# TAB 1: INSPECTION & SANITY CHECKS
# ==========================================
with tab1:
    st.header("Data Inspection & Sanity Checks")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Unique Users", f"{df['user id'].nunique():,}")
    col3.metric("Duplicate Users", f"{df['user id'].duplicated().sum()}")
    
    st.subheader("Group Sizes")
    group_counts = df['test group'].value_counts()
    group_pct = (group_counts / len(df) * 100).round(2)
    st.dataframe(pd.DataFrame({'Count': group_counts, 'Percentage (%)': group_pct}))

    st.subheader("Overall Conversions")
    total_conversions = df['converted'].sum()
    conversion_rate = df['converted'].mean()
    
    col4, col5 = st.columns(2)
    col4.metric("Total Converted Users", f"{total_conversions:,}")
    col5.metric("Overall Conversion Rate", f"{conversion_rate:.2%}")
    
    st.write("**Group Summary:**")
    group_summary = df.groupby('test group')['converted'].agg(
        users='count', conversions='sum', conversion_rate='mean'
    )
    group_summary['conversion_rate'] *= 100
    st.dataframe(group_summary.style.format({'conversion_rate': '{:.2f}%'}))

# ==========================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
with tab2:
    st.header("Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Ads - Overall")
        st.dataframe(df['total ads'].describe())
    with col2:
        st.subheader("Total Ads - By Group")
        st.dataframe(df.groupby('test group')['total ads'].describe())

    st.subheader("Visualizations")
    col3, col4 = st.columns(2)
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    with col3:
        conversion_by_day = df.groupby('most ads day')['converted'].mean().reindex(day_order) * 100
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(conversion_by_day.index, conversion_by_day.values)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in conversion_by_day.values], padding=3)
        ax.set_xlabel('Day')
        ax.set_ylabel('Conversion Rate (%)')
        ax.set_title('Conversion Rate by Day')
        ax.set_ylim(0, conversion_by_day.max() + 0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col4:
        conversion_by_hour = df.groupby('most ads hour')['converted'].mean() * 100
        x = conversion_by_hour.index.values
        y = conversion_by_hour.values
        
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)
        
        fig2 = plt.figure(figsize=(12, 5))
        plt.plot(x_smooth, y_smooth, linewidth=2)
        plt.fill_between(x_smooth, y_smooth, alpha=0.2)
        plt.xlabel('Hour')
        plt.ylabel('Conversion Rate (%)')
        plt.title('Conversion Rate by Hour')
        plt.xticks(range(0, 24))
        plt.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2)

    st.markdown("""
    ### Exploratory Analysis
    - **Ad exposure distribution:** The distribution of `total ads` is nearly identical between the control and test groups...
    - **Right-skewed ad exposure:** `total ads` is heavily right-skewed. The median is only 13 ads, compared with a mean of approximately 24.8...
    - **Conversion rate by day:** Conversion rate varies across days of the week. It is highest on Monday and Tuesday at around 3%...
    - **Conversion rate by hour:** Conversion rate also varies significantly by hour. It is lowest during the overnight period...
    
    > **Important:** The day-of-week and hourly patterns represent overall conversion behavior across all users. They have not yet been separated by the `ad` and `psa` groups.
    """)

# ==========================================
# TAB 3: COVARIATE BALANCE
# ==========================================
with tab3:
    st.header("Randomization and Covariate Balance Check")
    
    ad_total = df[df['test group'] == 'ad']['total ads']
    psa_total = df[df['test group'] == 'psa']['total ads']
    
    st.subheader("1. Compare total ads distribution between groups")
    u_stat, p_mw = stats.mannwhitneyu(ad_total, psa_total, alternative='two-sided')
    st.write(f"**Mann-Whitney U test p-value:** `{p_mw:.4f}`")
    
    st.subheader("2. Effect size & Cramér's V")
    def cramers_v(contingency_table):
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        rows, columns = contingency_table.shape
        return np.sqrt(chi2 / (n * min(rows - 1, columns - 1)))

    day_ct = pd.crosstab(df['test group'], df['most ads day'])
    hour_ct = pd.crosstab(df['test group'], df['most ads hour'])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Median total ads (Ad / PSA):** {ad_total.median():.1f} / {psa_total.median():.1f}")
        st.write(f"**Mean total ads (Ad / PSA):** {ad_total.mean():.2f} / {psa_total.mean():.2f}")
    with col2:
        st.write(f"**Cramér's V (Day vs Group):** {cramers_v(day_ct):.4f}")
        st.write(f"**Cramér's V (Hour vs Group):** {cramers_v(hour_ct):.4f}")

    st.markdown("""
    ### Interpretation and Conclusion
    **Interpretation:** All three p-values are extremely small; however, this is largely a consequence of the very large sample size (~588K rows). With such a large dataset, even very small differences can become statistically significant.
    
    The **effect sizes** provide a more useful perspective. Cramér's V is close to zero... indicating negligible associations. 
    
    **Conclusion:** There is no evidence of a **meaningfully broken or biased randomization** based on these observed covariates. The groups are well-balanced.
    """)

# ==========================================
# TAB 4: SIGNIFICANCE TESTING
# ==========================================
with tab4:
    st.header("Significance Testing")
    
    counts = df.groupby('test group')['converted'].agg(['sum', 'count'])
    x_ad, n_ad = counts.loc['ad', 'sum'], counts.loc['ad', 'count']
    x_psa, n_psa = counts.loc['psa', 'sum'], counts.loc['psa', 'count']
    p_ad, p_psa = x_ad / n_ad, x_psa / n_psa
    
    col1, col2 = st.columns(2)
    col1.metric("Ad Conversion Rate", f"{p_ad:.4%}")
    col2.metric("PSA Conversion Rate", f"{p_psa:.4%}")

    count, nobs = [x_ad, x_psa], [n_ad, n_psa]
    z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')
    ci_low, ci_high = confint_proportions_2indep(x_ad, n_ad, x_psa, n_psa, compare='diff', method='wald')
    
    st.subheader("Statistical Results")
    st.write(f"- **Observed Z-statistic:** `{z_stat:.4f}`")
    st.write(f"- **One-sided p-value:** `{p_value:.2e}`")
    st.write(f"- **Absolute Difference (Ad - PSA):** `{(p_ad - p_psa) * 100:.3f} pp`")
    st.write(f"- **95% Confidence Interval:** `[{ci_low * 100:.3f} pp, {ci_high * 100:.3f} pp]`")
    
    # Null Distribution Plot
    alpha = 0.05
    z_critical = stats.norm.ppf(1 - alpha)
    z_max = max(5, z_stat + 1)
    z = np.linspace(-4, z_max, 3000)
    pdf = stats.norm.pdf(z)
    
    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(z, pdf, linewidth=2, label='Null distribution')
    z_pvalue = z[z >= z_stat]
    plt.fill_between(z_pvalue, stats.norm.pdf(z_pvalue), 0, alpha=0.5, label=f'p-value region (p = {p_value:.2e})')
    z_rejection = z[z >= z_critical]
    plt.fill_between(z_rejection, stats.norm.pdf(z_rejection), 0, alpha=0.15, label=f'Rejection region (α = {alpha})')
    plt.axvline(z_stat, linestyle='--', linewidth=2, label=f'Observed Z = {z_stat:.2f}')
    plt.axvline(z_critical, linestyle=':', linewidth=2, label=f'Critical Z = {z_critical:.2f}')
    
    plt.xlabel('Z-statistic')
    plt.ylabel('Probability Density')
    plt.title('Two-Proportion Z-Test: Null Distribution\nAd Group vs PSA Group')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("""
    ### Verdict: Statistically Significant
    The hypothesis test provides **strong evidence of a difference in conversion rates**. Since the p-value is far below $\alpha = 0.05$, we **reject the null hypothesis**. 
    
    The estimated effect is both **statistically significant and practically meaningful**, corresponding to approximately a **43% relative lift in conversion rate** for the `ad` group.
    """)

# ==========================================
# TAB 5: BUSINESS IMPACT
# ==========================================
with tab5:
    st.header("Business Impact")
    
    diff = p_ad - p_psa
    incremental_point = diff * n_ad
    incremental_low = ci_low * n_ad
    incremental_high = ci_high * n_ad
    
    st.markdown(f"""
    ### 1. Incremental Conversions
    Across the {n_ad:,.0f} users in the `ad` group, the treatment generated:
    * **Point estimate:** ~{incremental_point:,.0f} conversions
    * **95% CI:** {incremental_low:,.0f} – {incremental_high:,.0f} additional conversions
    """)
    
    st.subheader("2. Revenue Sensitivity Analysis")
    margins = [20, 50, 100, 200]
    rev_data = []
    for m in margins:
        rev_data.append({
            "Margin/Conversion": f"${m}",
            "Point-Estimate Revenue": f"${incremental_point * m:,.0f}",
            "95% CI Revenue Range": f"${incremental_low * m:,.0f} - ${incremental_high * m:,.0f}"
        })
    st.table(pd.DataFrame(rev_data))
    
    st.subheader("3. Variation in Treatment Effect by Day")
    day_data = df.groupby(['most ads day', 'test group'])['converted'].agg(['sum', 'count']).reindex(day_order, level=0)
    
    lift_data = []
    for day in day_order:
        a_rate = day_data.loc[(day, 'ad'), 'sum'] / day_data.loc[(day, 'ad'), 'count']
        p_rate = day_data.loc[(day, 'psa'), 'sum'] / day_data.loc[(day, 'psa'), 'count']
        lift_data.append({
            "Day": day, 
            "Ad Conversion Rate": f"{a_rate*100:.2f}%", 
            "PSA Conversion Rate": f"{p_rate*100:.2f}%", 
            "Lift": f"{(a_rate - p_rate)*100:+.2f}pp"
        })
    st.table(pd.DataFrame(lift_data))
    
    st.markdown("""
    ### 4. Stakeholder Summary
    > Replacing the PSA with the ad increased conversion by **0.77 percentage points** (95% CI: **0.60–0.94pp**), corresponding to approximately a **43% relative lift** and a statistically significant result ($p < 0.001$). 
    """)