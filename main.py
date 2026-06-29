import pandas as pd

from data_pipeline import DataPipeline
from compliance_engine import run_compliance_check
from pricing_model import (
    run_pricing_model,
    engineer_features,
)
from win_probability import calculate_win_probability


def build_prediction_row(
    metadata: dict,
    requirements: dict,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a single-row dataframe for predicting the new tender.
    Missing historical values are replaced with dataset averages.
    """

    row = {

        "quantity":
            metadata.get("quantity", 1),

        "bid_validity_days":
            requirements.get("delivery_days", 30),

        "turnover_required_lakhs":
            requirements.get("min_turnover_lakhs", 0),

        "experience_required_years":
            requirements.get("min_experience_years", 0),

        "total_participants":
            historical_df["total_participants"].mean(),

        "qualified_bidders":
            historical_df["qualified_bidders"].mean(),

        "disqualified_bidders":
            historical_df["disqualified_bidders"].mean(),

        "qualified_mse":
            historical_df["qualified_mse"].mean(),

        "qualified_mii":
            historical_df["qualified_mii"].mean(),

        "num_bidders":
            historical_df["num_bidders"].mean(),

        "price_spread_pct":
            historical_df["price_spread_pct"].mean(),

        "product_category":
            metadata.get("product_category", "Unknown"),

        "ministry": historical_df["ministry"].mode()[0],

"department": historical_df["department"].mode()[0],

"organisation": historical_df["organisation"].mode()[0],

"state": historical_df["state"].mode()[0],
    }

    df = pd.DataFrame([row])

    df = engineer_features(df)

    return df


def main():

    print("=" * 80)
    print("GeM Tender Intelligence Engine")
    print("=" * 80)

    ###########################################################
    # Component A
    ###########################################################

    pipeline = DataPipeline().run()

    print("\nAvailable Tenders\n")

    tenders = pipeline.get_tender_names()

    for i, tender in enumerate(tenders):

        print(f"{i+1}. {tender}")

    choice = int(input("\nSelect Tender Number : "))

    tender_name = tenders[choice - 1]

    tender_text = pipeline.get_tender_text(tender_name)

    ###########################################################
    # Component B
    ###########################################################

    print("\nRunning Compliance Engine...\n")

    compliance = run_compliance_check(tender_text)

    requirements = compliance["requirements"]

    metadata = compliance["metadata"]

    vendors = compliance["vendors"]

    ###########################################################
    # Component C
    ###########################################################

    print("\nRunning Pricing Model...\n")

    pricing = run_pricing_model(pipeline.aoc_df)

    model = pricing["model"]

    prediction_df = build_prediction_row(
        metadata,
        requirements,
        pipeline.aoc_df,
    )

    predicted_price = model.predict(prediction_df)[0]

    print(f"\nPredicted L1 Price : ₹{predicted_price:,.2f}")

    ###########################################################
    # Component D
    ###########################################################

    vendor_bid = float(
    input("\nEnter your vendor bid : ₹")
    .replace(",", "")
    .strip()
)

    probability = calculate_win_probability(

        vendor_bid=vendor_bid,

        predicted_l1=predicted_price,

        num_competitors=int(
            prediction_df["num_bidders"].iloc[0]
        ),

        is_mse=True,

        is_mii=False,
    )

    ###########################################################
    # Final Report
    ###########################################################

    print("\n")
    print("=" * 80)
    print("GeM Tender Intelligence Report")
    print("=" * 80)

    print("\nTender")

    print(f"File              : {tender_name}")

    print(f"Quantity          : {metadata['quantity']}")

    print("\nProduct Requirements")

    if requirements["product_specs"]:

        for spec in requirements["product_specs"]:
            print(f"  • {spec}")

    else:

        print("  • No explicit technical specifications extracted.")

    print("\nCompliance Results")

    for vendor in vendors:

        print("-" * 60)

        print(vendor["vendor_name"])

        print(vendor["summary"])

    print("\nPricing Model")
    print("-" * 60)

    print(f"Predicted Winning Price : ₹{predicted_price:,.2f}")

    print("\nModel Confidence")

    print(f"Training Records   : {pricing['rows_used']}")

    print(f"Outliers Removed   : {pricing['outliers_removed']}")

    print(f"MAE               : ₹{pricing['metrics']['mae']:,.0f}")

    print(f"RMSE              : ₹{pricing['metrics']['rmse']:,.0f}")

    print(f"R²                : {pricing['metrics']['r2']:.3f}")

    print("\nTop Factors Used")

    top = pricing["feature_importance"].head(5)

    for _, row in top.iterrows():

        print(
            f"• {row['feature']} "
            f"({row['importance']:.3f})"
        )

    price_difference = (
        (vendor_bid - predicted_price)
        / predicted_price
    ) * 100

    print("\nWin Probability")
    print("-" * 60)

    print(f"Vendor Bid         : ₹{vendor_bid:,.2f}")

    print(f"Predicted L1       : ₹{predicted_price:,.2f}")

    print(f"Price Difference   : {price_difference:+.2f}%")

    print(f"Win Probability    : {probability['win_probability_pct']}%")

    print(f"Verdict            : {probability['verdict'].upper()}")

    print("\nOverall Summary")
    print("-" * 60)

    eligible = sum(v["is_eligible"] for v in vendors)

    print(f"Eligible Vendors      : {eligible}/{len(vendors)}")

    print(f"Estimated L1 Price    : ₹{predicted_price:,.2f}")

    print(f"Your Win Probability  : {probability['win_probability_pct']}%")

    print("=" * 80)


if __name__ == "__main__":
    main()