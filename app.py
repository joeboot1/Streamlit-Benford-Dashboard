"""
Benford's Law Analysis Dashboard
--------------------------------
Upload a CSV or Excel file, filter it, run a Benford analysis on the
first digit and the first two digits of a numeric column, test the
result against Benford's distribution with a Kolmogorov-Smirnov (KS)
statistic computed in NumPy, and download the results.
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Benford's Law Dashboard", page_icon="📊", layout="wide")
st.title("📊 Benford's Law Analysis Dashboard")
st.write(
    "Upload a CSV or Excel file, filter the data, and test whether a numeric "
    "column follows **Benford's Law** for the first digit and first two digits."
)

# ------------------------------------------------------------------
# Benford helper functions
# ------------------------------------------------------------------
def benford_expected(digits):
    """Benford's expected proportion for each leading digit (or digit pair)."""
    digits = np.asarray(digits, dtype=float)
    return np.log10(1 + 1 / digits)


def leading_digits(values, n_digits):
    """Return the first n significant digits of each value as integers.

    Scientific notation is used so that 0.0457 -> 4 (or 45) and
    4,570 -> 4 (or 45), and so floating-point rounding can't produce a bad digit.
    """
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values) & (values > 0)]
    sci = pd.Series(values).map(lambda v: f"{v:.14e}")  # e.g. '4.57000000000000e+03'
    if n_digits == 1:
        return sci.str[0].astype(int).to_numpy()
    return (sci.str[0] + sci.str[2]).astype(int).to_numpy()


def benford_table(values, n_digits):
    """Build a table of actual vs. Benford counts and proportions."""
    digits = np.arange(1, 10) if n_digits == 1 else np.arange(10, 100)
    observed = leading_digits(values, n_digits)
    n = len(observed)

    counts = np.array([np.sum(observed == d) for d in digits])
    actual_prop = counts / n if n > 0 else np.zeros(len(digits))
    expected_prop = benford_expected(digits)

    return pd.DataFrame(
        {
            "Digit": digits,
            "Actual Count": counts,
            "Expected Count": np.round(expected_prop * n, 2),
            "Actual %": actual_prop,
            "Benford %": expected_prop,
            "Difference": actual_prop - expected_prop,
        }
    ), n


# Critical-value coefficients for the KS test (c(alpha) / sqrt(n))
KS_COEFFICIENTS = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}


def ks_test_numpy(table, n, alpha):
    """KS test using NumPy: compare the cumulative actual distribution with
    the cumulative Benford distribution.

    KS statistic  D = max |CDF_actual - CDF_benford|
    Critical value  = c(alpha) / sqrt(n)
    If D > critical value, the data is statistically different from Benford.
    """
    cdf_actual = np.cumsum(table["Actual %"].to_numpy())
    cdf_benford = np.cumsum(table["Benford %"].to_numpy())
    abs_diff = np.abs(cdf_actual - cdf_benford)

    ks_stat = np.max(abs_diff)
    max_at = table["Digit"].to_numpy()[np.argmax(abs_diff)]
    critical = KS_COEFFICIENTS[alpha] / np.sqrt(n) if n > 0 else np.nan
    different = bool(ks_stat > critical)

    return {
        "KS Statistic (D)": round(float(ks_stat), 5),
        "Critical Value": round(float(critical), 5),
        "Significance Level (alpha)": alpha,
        "Sample Size (n)": int(n),
        "Largest Gap at Digit": int(max_at),
        "Statistically Different from Benford?": "Yes" if different else "No",
        "Conclusion": (
            "Reject H0: the distribution IS statistically different from Benford's Law."
            if different
            else "Fail to reject H0: the distribution is NOT statistically different from Benford's Law."
        ),
    }, cdf_actual, cdf_benford


def bar_chart(table, title):
    fig = go.Figure()
    fig.add_bar(x=table["Digit"], y=table["Actual %"], name="Actual", marker_color="#2a6fdb")
    fig.add_scatter(
        x=table["Digit"], y=table["Benford %"], name="Benford Expected",
        mode="lines+markers", line=dict(color="#e4572e", width=2),
    )
    fig.update_layout(
        title=title, xaxis_title="Leading Digit(s)", yaxis_title="Proportion",
        yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1), height=430,
    )
    return fig


def cdf_chart(table, cdf_actual, cdf_benford, title):
    fig = go.Figure()
    fig.add_scatter(x=table["Digit"], y=cdf_actual, name="Actual CDF",
                    mode="lines+markers", line=dict(color="#2a6fdb", shape="hv"))
    fig.add_scatter(x=table["Digit"], y=cdf_benford, name="Benford CDF",
                    mode="lines+markers", line=dict(color="#e4572e", shape="hv"))
    fig.update_layout(
        title=title, xaxis_title="Leading Digit(s)", yaxis_title="Cumulative Proportion",
        yaxis_tickformat=".0%", legend=dict(orientation="h", y=1.1), height=430,
    )
    return fig


# ------------------------------------------------------------------
# 1. Upload a CSV or Excel file
# ------------------------------------------------------------------
st.sidebar.header("1. Upload Data")
uploaded = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])


@st.cache_data
def load_file(file_bytes, name, sheet=None):
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)


if uploaded is None:
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

file_bytes = uploaded.getvalue()
sheet = None
if not uploaded.name.lower().endswith(".csv"):
    sheets = pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    sheet = st.sidebar.selectbox("Excel sheet", sheets)

df = load_file(file_bytes, uploaded.name, sheet)

# Turn text columns that look like dates into real dates (for date filtering)
for col in df.select_dtypes(include=["object", "string"]).columns:
    if "date" in col.lower():
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().mean() > 0.8:
            df[col] = converted

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if not numeric_cols:
    st.error("This file has no numeric columns to analyze. Please upload a different file.")
    st.stop()

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"**{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(100), width="stretch")

# ------------------------------------------------------------------
# 2. Choose the column of interest + filters
# ------------------------------------------------------------------
st.sidebar.header("2. Column of Interest")
target = st.sidebar.selectbox("Numeric column to analyze", numeric_cols)

st.sidebar.header("3. Filters")
filtered = df.copy()
active_filters = []

# Filter A: value range of the column of interest
col_min, col_max = float(df[target].min()), float(df[target].max())
if col_min < col_max:
    lo, hi = st.sidebar.slider(
        f"Range of {target}", min_value=col_min, max_value=col_max, value=(col_min, col_max)
    )
    filtered = filtered[filtered[target].between(lo, hi)]
    if (lo, hi) != (col_min, col_max):
        active_filters.append(f"{target} between {lo:,.2f} and {hi:,.2f}")

# Filter B: minimum absolute value (Benford tests usually exclude small numbers)
min_abs = st.sidebar.number_input(
    "Exclude values with absolute value below", min_value=0.0, value=10.0, step=1.0,
    help="Values under 10 have no second digit, so they're usually excluded for the first-two-digits test.",
)
filtered = filtered[filtered[target].abs() >= min_abs]
if min_abs > 0:
    active_filters.append(f"|{target}| ≥ {min_abs:,.2f}")

# Filter C: positive / negative / both
sign = st.sidebar.radio("Include values that are", ["Positive and negative", "Positive only", "Negative only"])
if sign == "Positive only":
    filtered = filtered[filtered[target] > 0]
    active_filters.append(f"{target} > 0")
elif sign == "Negative only":
    filtered = filtered[filtered[target] < 0]
    active_filters.append(f"{target} < 0")

# Filters D+: any other columns the user wants to filter by
other_cols = [c for c in df.columns if c != target]
extra = st.sidebar.multiselect("Filter by other columns", other_cols)
for col in extra:
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        start, end = series.min().date(), series.max().date()
        picked = st.sidebar.date_input(f"{col} date range", value=(start, end),
                                       min_value=start, max_value=end)
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            filtered = filtered[filtered[col].dt.date.between(picked[0], picked[1])]
            active_filters.append(f"{col} from {picked[0]} to {picked[1]}")
    elif pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
        cmin, cmax = float(series.min()), float(series.max())
        if cmin < cmax:
            r = st.sidebar.slider(f"{col} range", cmin, cmax, (cmin, cmax))
            filtered = filtered[filtered[col].between(*r)]
            active_filters.append(f"{col} between {r[0]:,.2f} and {r[1]:,.2f}")
    else:
        options = sorted(series.dropna().astype(str).unique().tolist())
        chosen = st.sidebar.multiselect(f"{col} values", options, default=options)
        filtered = filtered[filtered[col].astype(str).isin(chosen)]
        active_filters.append(f"{col} in {chosen if len(chosen) <= 5 else str(len(chosen)) + ' selected values'}")

st.sidebar.header("4. Test Settings")
alpha = st.sidebar.selectbox("Significance level (alpha)", [0.05, 0.01, 0.10], index=0)

values = filtered[target].dropna()
st.subheader("Filtered Data")
c1, c2, c3 = st.columns(3)
c1.metric("Rows in file", f"{len(df):,}")
c2.metric("Rows after filters", f"{len(filtered):,}")
c3.metric(f"Usable values in {target}", f"{(values != 0).sum():,}")
st.caption("Active filters: " + ("; ".join(active_filters) if active_filters else "none"))

if (values != 0).sum() < 10:
    st.warning("Fewer than 10 usable values remain. Loosen the filters to run the analysis.")
    st.stop()

# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------
st.subheader(f"Summary Statistics — {target}")
summary = values.describe().to_frame(name=target)
summary.loc["skew"] = values.skew()
summary.loc["sum"] = values.sum()
st.dataframe(summary.style.format("{:,.2f}"), width="stretch")

# ------------------------------------------------------------------
# 3. Benford analysis + KS test
# ------------------------------------------------------------------
table1, n1 = benford_table(values, 1)
table2, n2 = benford_table(values, 2)
ks1, cdf_a1, cdf_b1 = ks_test_numpy(table1, n1, alpha)
ks2, cdf_a2, cdf_b2 = ks_test_numpy(table2, n2, alpha)

fig1 = bar_chart(table1, f"First Digit: Actual vs. Benford ({target})")
fig1_cdf = cdf_chart(table1, cdf_a1, cdf_b1, "First Digit: Cumulative Distributions (KS Test)")
fig2 = bar_chart(table2, f"First Two Digits: Actual vs. Benford ({target})")
fig2_cdf = cdf_chart(table2, cdf_a2, cdf_b2, "First Two Digits: Cumulative Distributions (KS Test)")

pct = {"Actual %": "{:.2%}", "Benford %": "{:.2%}", "Difference": "{:+.2%}", "Expected Count": "{:,.1f}"}


def show_results(ks, table, fig, fig_cdf):
    m1, m2, m3 = st.columns(3)
    m1.metric("KS Statistic (D)", f"{ks['KS Statistic (D)']:.4f}")
    m2.metric(f"Critical Value (α = {alpha})", f"{ks['Critical Value']:.4f}")
    m3.metric("Different from Benford?", ks["Statistically Different from Benford?"])
    if ks["Statistically Different from Benford?"] == "Yes":
        st.error(ks["Conclusion"])
    else:
        st.success(ks["Conclusion"])
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(fig_cdf, width="stretch")
    with st.expander("Digit table"):
        st.dataframe(table.style.format(pct), width="stretch", hide_index=True)


st.subheader("Benford's Law Analysis")
tab1, tab2, tab3 = st.tabs(["First Digit", "First Two Digits", "Statistical Results"])
with tab1:
    show_results(ks1, table1, fig1, fig1_cdf)
with tab2:
    show_results(ks2, table2, fig2, fig2_cdf)
with tab3:
    results = pd.DataFrame({"First Digit": ks1, "First Two Digits": ks2})
    st.dataframe(results.astype(str), width="stretch")
    st.markdown(
        f"""
**How the test works**
- **H0:** the leading digits of *{target}* follow Benford's distribution.
- **KS statistic (D)** = the largest absolute gap between the cumulative actual
  distribution and the cumulative Benford distribution, computed with NumPy
  (`np.max(np.abs(np.cumsum(actual) - np.cumsum(benford)))`).
- **Critical value** = {KS_COEFFICIENTS[alpha]} / √n at α = {alpha}.
- If **D > critical value**, reject H0: the data is statistically different from Benford.
"""
    )

# ------------------------------------------------------------------
# 4. Downloads
# ------------------------------------------------------------------
st.subheader("Download & Share")
stamp = datetime.now().strftime("%Y-%m-%d_%H%M")

# Excel workbook with every table
excel_buf = io.BytesIO()
with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Summary Statistics")
    table1.to_excel(writer, sheet_name="First Digit", index=False)
    table2.to_excel(writer, sheet_name="First Two Digits", index=False)
    results.to_excel(writer, sheet_name="KS Test Results")
    filtered.to_excel(writer, sheet_name="Filtered Data", index=False)

# Shareable HTML report with interactive charts
report_html = f"""<html><head><meta charset="utf-8"><title>Benford Report - {target}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:20px}}
table{{border-collapse:collapse;margin-bottom:20px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}</style>
</head><body>
<h1>Benford's Law Analysis Report</h1>
<p><b>File:</b> {uploaded.name} &nbsp; <b>Column:</b> {target} &nbsp; <b>Generated:</b> {stamp}</p>
<p><b>Filters:</b> {"; ".join(active_filters) if active_filters else "none"}</p>
<h2>Summary Statistics</h2>{summary.round(2).to_html()}
<h2>KS Test Results</h2>{results.to_html()}
<h2>First Digit</h2>
{fig1.to_html(full_html=False, include_plotlyjs="cdn")}
{fig1_cdf.to_html(full_html=False, include_plotlyjs=False)}
{table1.round(4).to_html(index=False)}
<h2>First Two Digits</h2>
{fig2.to_html(full_html=False, include_plotlyjs=False)}
{fig2_cdf.to_html(full_html=False, include_plotlyjs=False)}
{table2.round(4).to_html(index=False)}
</body></html>"""

d1, d2, d3, d4 = st.columns(4)
d1.download_button("📄 Summary stats (CSV)", summary.to_csv().encode(),
                   f"summary_statistics_{stamp}.csv", "text/csv")
d2.download_button("📈 Benford results (CSV)",
                   pd.concat([table1.assign(Test="First Digit"),
                              table2.assign(Test="First Two Digits")]).to_csv(index=False).encode(),
                   f"benford_results_{stamp}.csv", "text/csv")
d3.download_button("📊 Full workbook (Excel)", excel_buf.getvalue(), f"benford_analysis_{stamp}.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
d4.download_button("🌐 Report with charts (HTML)", report_html.encode(),
                   f"benford_report_{stamp}.html", "text/html")
st.caption("The HTML report keeps the charts interactive. Open it in any browser or email it to share.")
