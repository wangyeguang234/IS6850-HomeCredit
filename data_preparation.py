"""
IS 6850 – Data Preparation Script
Home Credit Default Risk

Author: Yeguang Wang
Spring 2026

This script performs:
- Data cleaning
- Feature engineering
- Supplementary data aggregation
- Train/test consistent transformations

The output is modeling-ready datasets.
"""

import pandas as pd
import numpy as np


# ----------------------------------------------------------
# 1. CLEANING FUNCTIONS
# ----------------------------------------------------------

def clean_application(df):
    """
    Clean raw application data.
    """

    df = df.copy()

    # Fix DAYS_EMPLOYED anomaly
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    return df


# ----------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------

def engineer_features(df):
    """
    Create engineered features based on domain knowledge.
    """

    df = df.copy()

    # Age in years
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    # Employment duration in years
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365

    # Credit to income ratio
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    # Annuity to income ratio
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # Loan to annuity ratio
    df["LOAN_TO_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    # EXT_SOURCE average score
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    existing_ext = [col for col in ext_cols if col in df.columns]
    df["EXT_SOURCE_MEAN"] = df[existing_ext].mean(axis=1)

    return df


# ----------------------------------------------------------
# 3. MISSING VALUE HANDLING (TRAIN ONLY STATISTICS)
# ----------------------------------------------------------

def fit_missing_imputation(train_df):
    """
    Compute training statistics for missing value imputation.
    This prevents data leakage.
    """

    medians = train_df.median(numeric_only=True)

    return medians


def apply_missing_imputation(df, medians):
    """
    Apply precomputed medians to fill missing values.
    """

    df = df.copy()
    df = df.fillna(medians)

    return df


# ----------------------------------------------------------
# 4. AGGREGATE SUPPLEMENTARY DATA
# ----------------------------------------------------------

def aggregate_bureau(bureau_df):
    """
    Aggregate bureau data to applicant level (SK_ID_CURR).
    """

    bureau_agg = bureau_df.groupby("SK_ID_CURR").agg({
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "DAYS_CREDIT": ["mean", "max"]
    })

    # Flatten column names
    bureau_agg.columns = [
        "_".join(col).upper() for col in bureau_agg.columns
    ]

    bureau_agg = bureau_agg.reset_index()

    return bureau_agg


def merge_bureau(app_df, bureau_df):
    """
    Merge aggregated bureau data.
    """

    bureau_agg = aggregate_bureau(bureau_df)

    df = app_df.merge(
        bureau_agg,
        on="SK_ID_CURR",
        how="left"
    )

    return df


# ----------------------------------------------------------
# 5. FULL PIPELINE
# ----------------------------------------------------------

def prepare_datasets(train_path, test_path, bureau_path):
    """
    Full data preparation pipeline.
    Ensures train/test consistency.
    """

    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    bureau = pd.read_csv(bureau_path)

    # Cleaning
    train = clean_application(train)
    test = clean_application(test)

    # Feature engineering
    train = engineer_features(train)
    test = engineer_features(test)

    # Merge supplementary data
    train = merge_bureau(train, bureau)
    test = merge_bureau(test, bureau)

    # Fit imputation on training data only
    medians = fit_missing_imputation(train)

    # Apply to both
    train = apply_missing_imputation(train, medians)
    test = apply_missing_imputation(test, medians)

    # Ensure identical columns
    test = test.reindex(columns=train.columns.drop("TARGET"), fill_value=0)

    return train, test


# ----------------------------------------------------------
# 6. MAIN EXECUTION
# ----------------------------------------------------------

if __name__ == "__main__":

    train_processed, test_processed = prepare_datasets(
        "application_train.csv",
        "application_test.csv",
        "bureau.csv"
    )

    train_processed.to_csv("train_processed.csv", index=False)
    test_processed.to_csv("test_processed.csv", index=False)

    print("Data preparation complete.")
