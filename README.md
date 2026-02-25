# IS6850-HomeCredit
IS6850-HomeCredit
# IS 6850 – Data Preparation
Home Credit Default Risk Project

## Overview

This script performs data cleaning and feature engineering for the Home Credit Default Risk dataset. The goal is to transform raw application and supplementary data into a modeling-ready dataset while ensuring reproducibility and train/test consistency.

The script follows CRISP-DM best practices and prepares identical transformations for both training and test data.

---

## What the Script Does

The data_preparation.py script performs the following steps:

1. Cleans application data
   - Replaces the DAYS_EMPLOYED anomaly value (365243)
   - Handles missing values in EXT_SOURCE variables
   - Fixes other data quality issues identified during EDA

2. Creates engineered features
   - Age and employment duration
   - Credit-to-income ratio
   - Loan-to-value ratio
   - Missing value indicators
   - Additional ratio-based features

3. Aggregates supplementary datasets
   - bureau.csv
   - previous_application.csv
   - installments_payments.csv

4. Merges aggregated features at the applicant level (SK_ID_CURR)

5. Ensures train/test consistency
   - Computes statistics using training data only
   - Applies identical transformations to test data
   - Produces matching feature columns

## How to Run the Script

1. Place the raw data files in the project directory:
   - application_train.csv
   - application_test.csv
   - bureau.csv
   - previous_application.csv
   - installments_payments.csv

2. Run the script from the command line:

   python data_preparation.py

3. The script will generate processed training and test datasets ready for modeling.
   
## Inputs and Outputs

### Inputs
- Raw application data (application_train.csv, application_test.csv)
- Supplementary datasets (bureau.csv, previous_application.csv, installments_payments.csv)

### Outputs
- Processed training dataset with engineered features
- Processed test dataset with identical feature transformations
- Modeling-ready datasets with no data leakage
