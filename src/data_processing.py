import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
from src.constants import CATEGORICAL_COLS, NUMERICAL_COLS,TARGET_COL, COLUMN_NAMES

try:
    from xverse import iv, WoETransformer
    _has_xverse = True
    logger.info("`xverse` library detected IV function and WoETransformer available.")
except Exception:
    iv = None
    WoETransformer = None
    _has_xverse = False
    logger.info("`xverse` library not found – falling back to custom implementation.")

try:
    from woe import WOEEncoder  # type: ignore
    _has_woe = True
except Exception:
    WOEEncoder = None
    _has_woe = False

# 1. Data Loading Functions

def load_raw_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    # if set(COLUMN_NAMES).issubset(df.columns):
    #     logger.debug("Raw CSV contains a header row; using existing column names.")
    # else:
    #     logger.debug("Raw CSV file has no header; applying known column names.")
    #     df = pd.read_csv(filepath, names=COLUMN_NAMES)
    # df[TARGET_COL] = (df[TARGET_COL] == 2).astype(int)
    logger.info(f"Loaded {len(df)} records with {df.shape[1]} columns.")
    return df

# 2. Extract feature
class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create aggregate customer features and datetime features.
    """

    def __init__(
        self,
        customer_col="CustomerId",
        amount_col="Amount",
        datetime_col="TransactionStartTime",
    ):
        self.customer_col = customer_col
        self.amount_col = amount_col
        self.datetime_col = datetime_col

    def fit(self, X, y=None):

        customer_stats = (
            X.groupby(self.customer_col)[self.amount_col]
            .agg(
                total_transaction_amount="sum",
                avg_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std",
            )
            .reset_index()
        )

        customer_stats["std_transaction_amount"] = (
            customer_stats["std_transaction_amount"]
            .fillna(0)
        )

        self.customer_stats_ = customer_stats

        return self

    def transform(self, X):

        X = X.copy()

        # -------------------------------------
        # Aggregate Features
        # -------------------------------------

        X = X.merge(
            self.customer_stats_,
            on=self.customer_col,
            how="left"
        )

        # -------------------------------------
        # Datetime Features
        # -------------------------------------

        X[self.datetime_col] = pd.to_datetime(
            X[self.datetime_col]
        )

        X["transaction_hour"] = (
            X[self.datetime_col].dt.hour
        )

        X["transaction_day"] = (
            X[self.datetime_col].dt.day
        )

        X["transaction_month"] = (
            X[self.datetime_col].dt.month
        )

        X["transaction_year"] = (
            X[self.datetime_col].dt.year
        )

        return X
# feature scaler
class FeatureScaler(BaseEstimator, TransformerMixin):

    def __init__(self, numerical_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.numerical_cols_ = self.numerical_cols or [
            col for col in X.columns
            if pd.api.types.is_numeric_dtype(X[col]) and col != TARGET_COL
        ]
        self.scaler_ = StandardScaler()
        if self.numerical_cols_:
            self.scaler_.fit(X[self.numerical_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.numerical_cols_:
            X[self.numerical_cols_] = self.scaler_.transform(X[self.numerical_cols_])
        return X
# 3. Missing Value Handling (Imputation)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_before = df.isnull().sum().sum()
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.debug(f"Imputed {col} with median={median_val:.2f}")
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.debug(f"Imputed {col} with mode={mode_val}")
    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after}")
    return df


# 4. Encode Categorical Variables
class Encoder:
    def __init__(self):
        self.encoder = OneHotEncoder(
            sparse_output=True,
            handle_unknown="ignore"
        )
    def fit(self, df, categorical_cols):
        self.categorical_cols = categorical_cols
        self.encoder.fit(df[categorical_cols])

    def transform(self, df):
        encoded = self.encoder.transform(
            df[self.categorical_cols]
        )
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded,
            columns=self.feature_names_,
            index=X.index,
        )
        return pd.concat(
            [
                df.drop(columns=self.categorical_cols),
                encoded_df
            ],
            axis=1
        )


    def fit_transform(self, df, categorical_cols):
        self.fit(df, categorical_cols)
        return self.transform(df)

# 5. Normalization


# Numerical columns to normalize
NUMERICAL_COLUMNS = [
    "Amount",
    "Value",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_year",
]

TARGET_COL = "FraudResult"


def normalize_features(
    df: pd.DataFrame,
    columns: list[str] = NUMERICAL_COLUMNS
) -> tuple[pd.DataFrame, MinMaxScaler]:
    logger.info("Starting feature normalization.")

    df_norm = df.copy()

    scaler = MinMaxScaler()

    existing_cols = [col for col in columns if col in df_norm.columns]

    df_norm[existing_cols] = scaler.fit_transform(
        df_norm[existing_cols]
    )

    logger.info(
        f"Normalized {len(existing_cols)} numerical features."
    )

    return df_norm, scaler


def save_scaler(scaler, filepath: str):
    """
    Save fitted scaler for inference.
    """
    import joblib

    joblib.dump(scaler, filepath)
    logger.info(f"Scaler saved to {filepath}")


def load_scaler(filepath: str):
    """
    Load previously fitted scaler.
    """
    import joblib

    scaler = joblib.load(filepath)
    logger.info(f"Scaler loaded from {filepath}")
    return scaler

# feature engineering with Woe and IV
class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        sparse_output: bool = False,
        handle_unknown: str = "ignore",
    ):
        self.categorical_cols = categorical_cols
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown

    def fit(self, X: pd.DataFrame, y=None):
        present_cols = [col for col in X.columns if col in (self.categorical_cols or X.columns)]
        self.categorical_cols_ = [
            col for col in present_cols if not pd.api.types.is_numeric_dtype(X[col]) and col != TARGET_COL
        ]
        self.encoder_ = OneHotEncoder(
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown,
        )
        self.encoder_.fit(X[self.categorical_cols_])
        self.feature_names_ = self.encoder_.get_feature_names_out(self.categorical_cols_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        encoded = self.encoder_.transform(X[self.categorical_cols_])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        X = X.drop(columns=self.categorical_cols_)
        X = pd.concat([X, encoded_df], axis=1)
        return X


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = TARGET_COL,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, float]:
    total_events = (df[target] == 1).sum()
    total_non_events = (df[target] == 0).sum()

    stats = (
        df.groupby(feature, observed=False)[target]
        .agg(
            events=lambda x: (x == 1).sum(),
            non_events=lambda x: (x == 0).sum(),
        )
        .reset_index()
    )

    stats["dist_events"] = (stats["events"] + epsilon) / (total_events + epsilon)
    stats["dist_non_events"] = (stats["non_events"] + epsilon) / (total_non_events + epsilon)
    stats["woe"] = np.log(stats["dist_events"] / stats["dist_non_events"])
    stats["iv_component"] = (stats["dist_events"] - stats["dist_non_events"]) * stats["woe"]
    iv_total = stats["iv_component"].sum()
    return stats, iv_total


def _iv_label(iv: float) -> str:
    if iv < 0.02:
        return "Useless"
    elif iv < 0.1:
        return "Weak"
    elif iv < 0.3:
        return "Medium"
    elif iv < 0.5:
        return "Strong"
    else:
        return "Very Strong"


def compute_all_iv(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = TARGET_COL,
) -> pd.DataFrame:

    features = features or CATEGORICAL_COLS + NUMERICAL_COLS

    results = []

    for feat in features:

        if feat not in df.columns or feat == target:
            continue

        col = df[feat]

        if feat in NUMERICAL_COLS:
            try:
                col = pd.qcut(
                    df[feat],
                    q=10,
                    duplicates="drop"
                )
            except Exception:
                col = pd.cut(
                    df[feat],
                    bins=5
                )

        temp_df = df.copy()
        temp_df[feat] = col

        _, iv_value = compute_woe_iv(
            temp_df,
            feat,
            target
        )

        results.append(
            {
                "feature": feat,
                "iv": iv_value,
            }
        )

    iv_df = (
        pd.DataFrame(results)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )

    iv_df["predictive_power"] = (
        iv_df["iv"]
        .apply(_iv_label)
    )

    return iv_df


def encode_woe(
    df: pd.DataFrame,
    features: List[str],
    target: str = TARGET_COL,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    try:
        if WOEEncoder is None:
            raise ImportError("`woe` library is not installed.")
        encoder = WOEEncoder(cols=features, target=target)
        transformed = encoder.fit_transform(df.copy())
        woe_maps: Dict[str, Dict] = {}
        for col in features:
            woe_maps[col] = encoder.mapping_[col]
        return transformed, woe_maps
    except Exception:
        logger.warning("Falling back to custom WoE encoding due to error")
        df_copy = df.copy()
        woe_maps: Dict[str, Dict] = {}
        for feat in features:
            if feat not in df_copy.columns:
                continue
            woe_df, _ = compute_woe_iv(df_copy, feat, target)
            woe_map = dict(zip(woe_df[feat], woe_df["woe"]))
            df_copy[f"{feat}_woe"] = df_copy[feat].map(woe_map)
            woe_maps[feat] = woe_map
        return df_copy, woe_maps


class WoEFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_cols: Optional[List[str]] = None, target_col: str = TARGET_COL):
        self.categorical_cols = categorical_cols
        self.target_col = target_col

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if y is not None:
            X[self.target_col] = y
        self.categorical_cols_ = self.categorical_cols or [
            col for col in X.columns
            if col not in [self.target_col] and not pd.api.types.is_numeric_dtype(X[col])
        ]
        self.woe_maps_ = {}
        for col in self.categorical_cols_:
            if col not in X.columns:
                continue
            woe_df, _ = compute_woe_iv(X[[col, self.target_col]],col,target=self.target_col)

            woe_df, _ = compute_woe_iv(
                X[[col, self.target_col]],
                col,
                target=self.target_col
            )

        self.woe_maps_[col] = dict(zip(woe_df[col], woe_df["woe"]))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.woe_maps_.items():
            if col in X.columns:
                X[f"{col}_woe"] = X[col].map(mapping).fillna(0.0)
        return X

def fit_transform_pipeline(
    df: pd.DataFrame,
    use_woe: bool = False,
) -> Tuple[Pipeline, pd.DataFrame]:
    pipeline = build_feature_pipeline(use_woe=use_woe)
    transformed = pipeline.fit_transform(df)
    return pipeline, transformed

def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline(raw_filepath: str, output_dir: str, use_woe: bool = False) -> Dict:
    """End-to-end pipeline: raw input -> model-ready CSV output."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_data(raw_filepath)
    pipeline = build_feature_pipeline(use_woe=use_woe)
    df_processed = pipeline.fit_transform(df)

    processed_path = os.path.join(output_dir, "german_credit_processed.csv")
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to: {processed_path}")

    return {
        "df": df_processed,
        "pipeline": pipeline,
        "processed_path": processed_path,
        "n_records": len(df_processed),
        "n_features": df_processed.shape[1] - 1,
        "default_rate": df_processed[TARGET_COL].mean(),
    }

def build_feature_pipeline():

    return Pipeline([
        ("engineer", TransactionFeatureEngineer()),  # needs CustomerId FIRST

        ("one_hot", DataFrameOneHotEncoder(
            categorical_cols=CATEGORICAL_COLS
        )),

        ("scale", FeatureScaler(
            numerical_cols=NUMERICAL_COLS + [
                "transaction_hour",
                "transaction_day",
                "transaction_month",
                "transaction_year",
            ]
        )),
    ])


def fit_transform_pipeline(
    df: pd.DataFrame,
    use_woe: bool = False,
) -> Tuple[Pipeline, pd.DataFrame]:
    pipeline = build_feature_pipeline(use_woe=use_woe)
    transformed = pipeline.fit_transform(df)
    return pipeline, transformed


def split_data(
    df: pd.DataFrame,
    target: str = TARGET_COL,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=random_state, stratify=y_temp
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_pipeline(raw_filepath: str, output_dir: str, use_woe: bool = False) -> Dict:
    """End-to-end pipeline: raw input -> model-ready CSV output."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_data(raw_filepath)
    pipeline = build_feature_pipeline(use_woe=use_woe)
    df_processed = pipeline.fit_transform(df)

    processed_path = os.path.join(output_dir, "credit_risk_processed.csv")
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to: {processed_path}")

    return {
        "df": df_processed,
        "pipeline": pipeline,
        "processed_path": processed_path,
        "n_records": len(df_processed),
        "n_features": df_processed.shape[1] - 1,
        "default_rate": df_processed[TARGET_COL].mean(),
    }







# RFM metrics

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_rfm_features(df, customer_col="CustomerId",
                        date_col="TransactionDate",
                        amount_col="TransactionAmount",
                        snapshot_date=None):
    """
    Compute RFM features per customer.
    """

    df = df.copy()

    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Define snapshot date (if not provided, use max transaction date + 1 day)
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm = df.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(date_col, "count"),
        Monetary=(amount_col, "sum")
    ).reset_index()

    return rfm


def cluster_customers_rfm(rfm, n_clusters=3, random_state=42):
    """
    Scale RFM and apply KMeans clustering.
    """

    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    return rfm, scaler, kmeans


def identify_high_risk_cluster(rfm):
    """
    Identify cluster with highest Recency and lowest Frequency/Monetary.
    """

    cluster_summary = rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()

    # High risk = high recency + low frequency + low monetary
    cluster_scores = (
        cluster_summary["Recency"]
        - cluster_summary["Frequency"]
        - cluster_summary["Monetary"]
    )

    high_risk_cluster = cluster_scores.idxmax()

    return high_risk_cluster


def add_high_risk_label(df, rfm, high_risk_cluster, customer_col="CustomerId"):
    """
    Merge cluster labels and create binary target.
    """

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    df_out = df.merge(
        rfm[[customer_col, "is_high_risk"]],
        on=customer_col,
        how="left"
    )

    return df_out


def build_rfm_pipeline(df,
                       customer_col="CustomerId",
                       date_col="TransactionDate",
                       amount_col="TransactionAmount",
                       snapshot_date=None,
                       n_clusters=3,
                       random_state=42):
    """
    Full pipeline: RFM -> clustering -> high-risk labeling -> merge back.
    """

    rfm = create_rfm_features(
        df,
        customer_col=customer_col,
        date_col=date_col,
        amount_col=amount_col,
        snapshot_date=snapshot_date
    )

    rfm, scaler, kmeans = cluster_customers_rfm(
        rfm,
        n_clusters=n_clusters,
        random_state=random_state
    )

    high_risk_cluster = identify_high_risk_cluster(rfm)

    df_out = add_high_risk_label(df, rfm, high_risk_cluster, customer_col)

    return df_out, rfm, scaler, kmeans, high_risk_cluster


if __name__ == "__main__":
    # Example usage (update path as needed)
    df = pd.read_csv("data/transactions.csv")

    processed_df, rfm_table, scaler, model, high_risk_cluster = build_rfm_pipeline(df)

    processed_df.to_csv("data/processed_dataset.csv", index=False)

    print("High-risk cluster:", high_risk_cluster)
    print("Saved processed dataset with is_high_risk column.")