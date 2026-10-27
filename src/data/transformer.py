"""Feature engineering and preprocessing pipeline for fraud detection.

Implements an efficient, unified feature extraction pipeline used both
for offline model training and for ultra-low latency online inference.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.schemas import TransactionInput

DEFAULT_IP_MAPPING_PATH: Path = Path("data/raw/IpAddress_to_Country.csv")
TOP_COUNTRIES_DEFAULT: int = 25


class IpCountryLookup:
    """Fast binary-search lookup table for numeric IP to ISO country mapping."""

    def __init__(self, ip_csv_path: Path = DEFAULT_IP_MAPPING_PATH) -> None:
        """Load and sort IP range table in memory.

        Args:
            ip_csv_path: Path to the IpAddress_to_Country CSV file.
        """
        self.ip_csv_path = ip_csv_path
        self._lower_bounds: np.ndarray = np.array([], dtype=np.float64)
        self._upper_bounds: np.ndarray = np.array([], dtype=np.float64)
        self._countries: np.ndarray = np.array([], dtype=object)
        self._is_loaded = False
        if ip_csv_path.exists():
            self._load()

    def _load(self) -> None:
        df = (
            pd.read_csv(self.ip_csv_path)
            .sort_values("lower_bound_ip_address")
            .reset_index(drop=True)
        )
        self._lower_bounds = df["lower_bound_ip_address"].values
        self._upper_bounds = df["upper_bound_ip_address"].values
        self._countries = df["country"].values
        self._is_loaded = True

    def lookup(self, ip_values: np.ndarray | list[float] | float) -> np.ndarray:
        """Map numeric IP addresses to country names via binary search.

        Complexity is O(log M) per lookup, achieving sub-millisecond execution.

        Args:
            ip_values: Single IP or array of float IP addresses.

        Returns:
            Numpy array of string country names.
        """
        if not self._is_loaded:
            if self.ip_csv_path.exists():
                self._load()
            else:
                return np.full(np.shape(ip_values), "Unknown", dtype=object)

        ips = np.asarray(ip_values, dtype=np.float64)
        scalar_input = ips.ndim == 0
        if scalar_input:
            ips = np.array([ips])

        idx = np.searchsorted(self._lower_bounds, ips, side="right") - 1
        clipped_idx = np.clip(idx, 0, len(self._upper_bounds) - 1)
        valid = (idx >= 0) & (ips <= self._upper_bounds[clipped_idx])
        result = np.where(
            valid, self._countries[np.clip(idx, 0, len(self._countries) - 1)], "Unknown"
        )

        return result[0] if scalar_input else result


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Unified scikit-learn transformer for fraud detection feature engineering."""

    def __init__(
        self,
        ip_mapping_path: Path = DEFAULT_IP_MAPPING_PATH,
        top_countries_count: int = TOP_COUNTRIES_DEFAULT,
    ) -> None:
        """Initialize transformer attributes.

        Args:
            ip_mapping_path: Path to the IP country mapping table.
            top_countries_count: Number of top country categories to retain.
        """
        self.ip_mapping_path = ip_mapping_path
        self.top_countries_count = top_countries_count
        self.ip_lookup = IpCountryLookup(ip_mapping_path)
        self.top_countries_: list[str] = []
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FraudFeatureTransformer":
        """Fit transformer by learning high-frequency countries and schema.

        Args:
            X: Input DataFrame with raw transactions.
            y: Optional target array (unused).

        Returns:
            Self instance.
        """
        countries = self.ip_lookup.lookup(X["ip_address"].values)
        country_counts = pd.Series(countries).value_counts()
        self.top_countries_ = list(
            country_counts[country_counts.index != "Unknown"].head(self.top_countries_count).index
        )
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw DataFrame into ML feature matrix.

        Args:
            df: DataFrame containing raw transaction columns.

        Returns:
            Processed DataFrame with engineered features.
        """
        res = pd.DataFrame(index=df.index)

        signup = pd.to_datetime(df["signup_time"])
        purchase = pd.to_datetime(df["purchase_time"])
        time_diff = (purchase - signup).dt.total_seconds()

        res["purchase_value"] = df["purchase_value"].astype(np.float32)
        res["age"] = df["age"].astype(np.int32)
        res["time_since_signup_sec"] = np.maximum(time_diff, 0.0).astype(np.float32)
        res["is_instant_purchase"] = (res["time_since_signup_sec"] < 5.0).astype(np.int32)
        res["hour_of_day"] = purchase.dt.hour.astype(np.int32)
        res["day_of_week"] = purchase.dt.dayofweek.astype(np.int32)

        # Categoricals
        res["sex_M"] = (df["sex"] == "M").astype(np.int32)

        for source in ["SEO", "Ads", "Direct"]:
            res[f"source_{source}"] = (df["source"] == source).astype(np.int32)

        for browser in ["Chrome", "Safari", "Firefox", "Opera", "IE"]:
            res[f"browser_{browser}"] = (df["browser"] == browser).astype(np.int32)

        # Country one-hot encoding
        countries = self.ip_lookup.lookup(df["ip_address"].values)
        for country in self.top_countries_:
            safe_name = country.replace(" ", "_").replace("'", "")
            res[f"country_{safe_name}"] = (countries == country).astype(np.int32)

        if not self.feature_names_:
            self.feature_names_ = list(res.columns)

        return res[self.feature_names_]

    def transform_single(self, tx: TransactionInput) -> np.ndarray:
        """Transform a single online inference request into a 2D float32 numpy array.

        Args:
            tx: Validated Pydantic TransactionInput instance.

        Returns:
            2D numpy array of shape (1, n_features) ready for ONNX runtime inference.
        """
        signup = tx.signup_time
        purchase = tx.purchase_time
        time_diff = max((purchase - signup).total_seconds(), 0.0)
        country = str(self.ip_lookup.lookup(tx.ip_address))

        feature_values: list[float] = [
            float(tx.purchase_value),
            float(tx.age),
            float(time_diff),
            1.0 if time_diff < 5.0 else 0.0,
            float(purchase.hour),
            float(purchase.weekday()),
            1.0 if tx.sex == "M" else 0.0,
            1.0 if tx.source == "SEO" else 0.0,
            1.0 if tx.source == "Ads" else 0.0,
            1.0 if tx.source == "Direct" else 0.0,
            1.0 if tx.browser == "Chrome" else 0.0,
            1.0 if tx.browser == "Safari" else 0.0,
            1.0 if tx.browser == "Firefox" else 0.0,
            1.0 if tx.browser == "Opera" else 0.0,
            1.0 if tx.browser == "IE" else 0.0,
        ]

        for c in self.top_countries_:
            feature_values.append(1.0 if country == c else 0.0)

        return np.array([feature_values], dtype=np.float32)

    def save(self, filepath: Path) -> None:
        """Serialize transformer instance to disk using joblib.

        Args:
            filepath: Destination file path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "FraudFeatureTransformer":
        """Load serialized transformer from disk.

        Args:
            filepath: Path to saved transformer joblib file.

        Returns:
            Loaded FraudFeatureTransformer instance.
        """
        return joblib.load(filepath)
