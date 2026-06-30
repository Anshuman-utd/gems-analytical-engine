import streamlit as st
import pandas as pd

from data_pipeline import DataPipeline
from compliance_engine import run_compliance_check
from pricing_model import (
    run_pricing_model,
    engineer_features,
)
from win_probability import calculate_win_probability

from main import build_prediction_row


st.set_page_config(
    page_title="GeM Tender Intelligence",
    page_icon="📑",
    layout="wide",
)

st.markdown("""
<style>

div[data-testid="metric-container"]{
    border:1px solid #e6e6e6;
    padding:18px;
    border-radius:12px;
    background-color:#fafafa;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

st.title("📑 GeM Tender Intelligence Dashboard")

st.markdown(
"""
Analyze Government tenders using AI-powered compliance checking,
machine learning price prediction and dynamic bid intelligence.
"""
)

st.divider()
st.caption(
    "AI-powered Tender Compliance, Price Prediction & Win Probability"
)

@st.cache_resource
def load_pipeline():

    return DataPipeline().run()

pipeline = load_pipeline()


st.sidebar.title("📊 GeM AI")

st.sidebar.markdown(
"""
Government e-Marketplace
Tender Intelligence System
"""
)

st.sidebar.divider()

st.sidebar.subheader("Tender Selection")

tenders = pipeline.get_tender_names()

selected_tender = st.sidebar.selectbox(
    "Select Tender",
    tenders,
)

tender_text = pipeline.get_tender_text(
    selected_tender
)

@st.cache_data
def get_compliance(text):

    return run_compliance_check(text)

compliance = get_compliance(tender_text)


vendors = compliance["vendors"]

vendor_names = [
    v["vendor_name"]
    for v in vendors
]

selected_vendor_name = st.sidebar.selectbox(
    "Select Vendor",
    vendor_names,
)

selected_vendor = next(
    v
    for v in vendors
    if v["vendor_name"] == selected_vendor_name
)


vendor_bid = st.sidebar.number_input(

    "Vendor Bid (₹)",

    min_value=1000.0,

    value=100000.0,

    step=1000.0,
)

st.sidebar.divider()

st.sidebar.info(
"""
Built using

• Llama 3 (Groq)

• Random Forest

• Streamlit

• Scikit-Learn
"""
)


@st.cache_resource
def load_pricing():
    return run_pricing_model(pipeline.aoc_df)


pricing = load_pricing()

prediction_df = build_prediction_row(
    compliance["metadata"],
    compliance["requirements"],
    pipeline.aoc_df,
)

predicted_price = pricing["model"].predict(prediction_df)[0]

probability = calculate_win_probability(
    vendor_bid=vendor_bid,
    predicted_l1=predicted_price,
    num_competitors=int(prediction_df["num_bidders"].iloc[0]),
    is_mse=True,
    is_mii=False,
)

# =====================================================
# TOP METRICS
# =====================================================

st.divider()

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Quantity",
    compliance["metadata"]["quantity"],
)

m2.metric(
    "Predicted L1",
    f"₹{predicted_price:,.0f}",
)

m3.metric(
    "Win Probability",
    f"{probability['win_probability_pct']}%",
)

m4.metric(
    "Vendor Status",
    "Eligible" if selected_vendor["is_eligible"] else "Ineligible",
)

st.divider()

# =====================================================
# MAIN DASHBOARD
# =====================================================

left, right = st.columns([1.3, 1])

# =====================================================
# LEFT
# =====================================================

with left:

    st.subheader("📋 Product Requirements")

    specs = compliance["requirements"]["product_specs"]

    if specs:
        for spec in specs:
            st.markdown(f"✅ **{spec}**")
    else:
        st.info("No explicit technical specifications extracted.")

    st.divider()

    st.subheader("💰 Pricing Model")

    p1, p2 = st.columns(2)
    p3, p4 = st.columns(2)

    p1.metric(
        "Training Rows",
        pricing["rows_used"],
    )

    p2.metric(
        "MAE",
        f"₹{pricing['metrics']['mae']:,.0f}",
    )

    p3.metric(
        "RMSE",
        f"₹{pricing['metrics']['rmse']:,.0f}",
    )

    p4.metric(
        "R² Score",
        f"{pricing['metrics']['r2']:.3f}",
    )

    st.divider()

    with st.expander("📈 Feature Importance", expanded=True):

        top = pricing["feature_importance"].head(10)

        top = top.iloc[::-1]

        st.bar_chart(
            top.set_index("feature")["importance"]
        )

# =====================================================
# RIGHT
# =====================================================

with right:

    st.subheader("✅ Compliance")

    if selected_vendor["is_eligible"]:
        st.success(selected_vendor["summary"])
    else:
        st.error(selected_vendor["summary"])

    checks = []

    for name, data in selected_vendor["checks"].items():

        status = "✅ PASS"

        if not data.get("pass", True):
            status = "❌ FAIL"

        row = {
            "Requirement": name.replace("_", " ").title(),
            "Status": status,
        }

        if "required" in data:
            row["Required"] = data["required"]

        if "vendor_has" in data:
            row["Vendor"] = data["vendor_has"]

        checks.append(row)

    with st.expander("Evaluation Matrix", expanded=True):

        st.dataframe(
            pd.DataFrame(checks),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader("🎯 Win Probability")

    difference = (
        (vendor_bid - predicted_price)
        / predicted_price
    ) * 100

    w1, w2 = st.columns(2)

    w1.metric(
        "Vendor Bid",
        f"₹{vendor_bid:,.0f}",
    )

    w2.metric(
        "Predicted L1",
        f"₹{predicted_price:,.0f}",
    )

    st.metric(
        "Price Difference",
        f"{difference:+.2f}%",
    )

    st.metric(
        "Probability",
        f"{probability['win_probability_pct']}%",
    )

    st.progress(
        probability["win_probability_pct"] / 100
    )

    verdict = probability["verdict"]

    if verdict in ["Very Strong", "Strong"]:
        st.success(verdict)

    elif verdict == "Moderate":
        st.warning(verdict)

    else:
        st.error(verdict)

# =====================================================
# VENDOR COMPARISON
# =====================================================

st.divider()

st.subheader("👥 Vendor Comparison")

table = []

for vendor in vendors:

    table.append({

        "Vendor": vendor["vendor_name"],

        "Eligible": "Yes" if vendor["is_eligible"] else "No",

        "Summary": vendor["summary"],

        "Selected": "⭐" if vendor["vendor_name"] == selected_vendor_name else "",

    })

st.dataframe(
    pd.DataFrame(table),
    use_container_width=True,
    hide_index=True,
)

# =====================================================
# EXECUTIVE SUMMARY
# =====================================================

st.divider()

st.subheader("📌 Executive Summary")

st.markdown(f"""
### Selected Vendor

**{selected_vendor["vendor_name"]}**

---

**Eligibility**

{"🟢 Eligible" if selected_vendor["is_eligible"] else "🔴 Not Eligible"}

---

**Predicted Winning Price**

**₹{predicted_price:,.0f}**

---

**Estimated Win Probability**

**{probability["win_probability_pct"]}%**

---

**Recommendation**

### {probability["verdict"]}
""")