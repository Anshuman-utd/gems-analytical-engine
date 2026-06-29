from data_pipeline import DataPipeline
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import numpy as np







logger = logging.getLogger(__name__)

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs ML-specific data sanitization.

    The DataPipeline already handles:
    - Duplicate removal
    - Datatype conversion
    - Basic validation

    This function prepares the dataset specifically for
    machine learning.

    Steps:
    1. Remove rows with missing target values
    2. Remove impossible numeric values
    3. Fill optional requirement columns
    4. Convert categorical NaNs to 'Unknown'
    """

    df = df.copy()

    logger.info(f"Starting dataset cleaning ({len(df)} rows)")

    # --------------------------------------------------
    # Target column must exist
    # --------------------------------------------------

    before = len(df)

    df = df.dropna(subset=["l1_price"])

    logger.info(
        f"Removed {before - len(df)} rows with missing L1 price"
    )

    # --------------------------------------------------
    # Remove impossible values
    # --------------------------------------------------

    before = len(df)

    df = df[
        (df["l1_price"] > 0)
        &
        (df["quantity"] > 0)
        &
        (df["num_bidders"] > 0)
    ]

    logger.info(
        f"Removed {before - len(df)} rows with invalid values"
    )

    # --------------------------------------------------
    # Fill optional numeric columns
    # Missing means "not required"
    # --------------------------------------------------

    numeric_fill_zero = [
        "turnover_required_lakhs",
        "experience_required_years",
        "qualified_mse",
        "qualified_mii",
        "disqualified_bidders",
        "price_spread_pct",
    ]

    for col in numeric_fill_zero:

        if col in df.columns:

            df[col] = df[col].fillna(0)

    # --------------------------------------------------
    # Fill optional categorical columns
    # --------------------------------------------------

    categorical_fill_unknown = [
        "product_category",
        "ministry",
        "department",
        "organisation",
        "office",
        "state",
    ]

    for col in categorical_fill_unknown:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("Unknown")
                .astype(str)
                .str.strip()
            )

    logger.info(
        f"Dataset ready for ML ({len(df)} rows)"
    )

    return df

def remove_outliers(
    df: pd.DataFrame,
    column: str = "l1_price"
) -> tuple[pd.DataFrame, int]:
    """
    Removes extreme price outliers using the IQR method.

    Why?
    GeM data can contain:
    - predatory pricing
    - accidental bids
    - abnormal high quotations

    These distort ML training.
    """

    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    clean_df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ].copy()

    removed = len(df) - len(clean_df)

    logger.info(
        f"Removed {removed} outlier bids "
        f"(allowed range ₹{lower:,.0f} – ₹{upper:,.0f})"
    )

    return clean_df, removed


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates additional features that improve prediction.
    """

    df = df.copy()

    # -----------------------------
    # Competition Ratio
    # -----------------------------

    df = df.copy()

    df["competition_ratio"] = (
        df["qualified_bidders"]
        /
        df["total_participants"].replace(0, 1)
    )

    if (
        "bid_start" in df.columns
        and
        "bid_end" in df.columns
    ):

        ...

    elif "bid_validity_days" in df.columns:

        df["bid_duration_hours"] = (
            df["bid_validity_days"] * 24
        )

    return df

def build_preprocessor():

    numeric_features = [

        "quantity",

        "bid_validity_days",

        "turnover_required_lakhs",

        "experience_required_years",

        "total_participants",

        "qualified_bidders",

        "disqualified_bidders",

        "qualified_mse",

        "qualified_mii",

        "num_bidders",

        "price_spread_pct",

        "competition_ratio",

        "bid_duration_hours",

    ]

    categorical_features = [

        "product_category",

        "ministry",

        "department",

        "organisation",

        "state",

    ]

    numeric_pipeline = Pipeline(

        [

            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

        ]

    )

    categorical_pipeline = Pipeline(

        [

            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),

            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),

        ]

    )

    preprocessor = ColumnTransformer(

        [

            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),

            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),

        ]

    )

    return preprocessor


def train_model(df: pd.DataFrame):

    target = "l1_price"

    feature_columns = [
    "quantity",
    "bid_validity_days",
    "turnover_required_lakhs",
    "experience_required_years",
    "total_participants",
    "qualified_bidders",
    "disqualified_bidders",
    "qualified_mse",
    "qualified_mii",
    "num_bidders",
    "price_spread_pct",
    "competition_ratio",
    "bid_duration_hours",
    "product_category",
    "ministry",
    "department",
    "organisation",
    "state",
    ]   

    X = df[feature_columns]

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42,

    )

    model = Pipeline(

        [

            (

                "preprocessor",

                build_preprocessor(),

            ),

            (

                "model",

                RandomForestRegressor(

                    n_estimators=300,

                    random_state=42,

                    max_depth=12,

                    min_samples_leaf=2,

                ),

            ),

        ]

    )

    model.fit(

        X_train,

        y_train,

    )

    feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.named_steps["model"].feature_importances_,
        }
    )

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False,
    )

    logger.info("Random Forest trained")

    return (

        model,

        X_test,

        y_test,

        feature_importance,

    )

def evaluate_model(

    model,

    X_test,

    y_test,

):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(

        y_test,

        predictions,

    )

    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            predictions,

        )

    )

    r2 = r2_score(

        y_test,

        predictions,

    )

    logger.info(f"MAE : ₹{mae:,.0f}")

    logger.info(f"RMSE: ₹{rmse:,.0f}")

    logger.info(f"R²  : {r2:.3f}")

    return {

        "mae": mae,

        "rmse": rmse,

        "r2": r2,

    }

def predict_l1(

    model,

    tender_features: pd.DataFrame,

):

    prediction = model.predict(

        tender_features

    )[0]

    return round(

        float(prediction),

        2,

    )

def run_pricing_model(df):

    df = clean_dataset(df)

    df, removed = remove_outliers(df)

    df = engineer_features(df)

    model, X_test, y_test, feature_importance = train_model(df)

    metrics = evaluate_model(

        model,

        X_test,

        y_test,

    )

 

    return {

        "model": model,

        "metrics": metrics,

        "rows_used": len(df),

        "outliers_removed": removed,

        "feature_importance": feature_importance,

    }

if __name__ == "__main__":

    pipeline = DataPipeline().run()

    df = pipeline.aoc_df.copy()

    results = run_pricing_model(df)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)
    print(f"Rows Used        : {results['rows_used']}")
    print(f"Outliers Removed : {results['outliers_removed']}")
    print(f"MAE              : ₹{results['metrics']['mae']:.2f}")
    print(f"RMSE             : ₹{results['metrics']['rmse']:.2f}")
    print(f"R² Score         : {results['metrics']['r2']:.4f}")