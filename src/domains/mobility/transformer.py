"""Feature engineering and preprocessing pipeline for urban mobility traffic forecasting.

Provides an efficient, leak-free feature extraction pipeline used both for offline
batch training and ultra-low latency online inference (ONNX compatible).
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.domains.mobility.schemas import TripInput

TOP_LOCATIONS_DEFAULT: int = 25
AIRPORT_LOCATION_IDS: set[int] = {1, 132, 138}  # EWR, JFK, LaGuardia


class MobilityFeatureTransformer(BaseEstimator, TransformerMixin):
    """Unified scikit-learn transformer for urban mobility feature engineering.

    Guarantees strict train/serve parity with zero data leakage.
    Categorical high-cardinality locations (265 zones) are encoded using Top-K one-hot
    vectors to ensure deterministic fixed-width float32 arrays for ONNX runtime serving.
    """

    def __init__(self, top_locations_count: int = TOP_LOCATIONS_DEFAULT) -> None:
        """Initialize transformer parameters.

        Args:
            top_locations_count: Number of most frequent pickup and dropoff zones to retain.
        """
        self.top_locations_count = top_locations_count
        self.top_pickup_locations_: list[int] = []
        self.top_dropoff_locations_: list[int] = []
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MobilityFeatureTransformer":
        """Learn top frequent pickup and dropoff zone IDs from training data.

        Args:
            X: Input DataFrame containing raw taxi trips (with PULocationID, DOLocationID).
            y: Optional target array (unused).

        Returns:
            Fitted transformer instance.
        """
        pu_counts = X["PULocationID"].value_counts()
        do_counts = X["DOLocationID"].value_counts()

        self.top_pickup_locations_ = [
            int(loc) for loc in pu_counts.head(self.top_locations_count).index
        ]
        self.top_dropoff_locations_ = [
            int(loc) for loc in do_counts.head(self.top_locations_count).index
        ]

        # Determine feature column names deterministically
        feature_names = [
            "trip_distance",
            "passenger_count",
            "pickup_hour",
            "pickup_dayofweek",
            "is_weekend",
            "is_airport_pickup",
            "is_airport_dropoff",
        ]
        for loc in self.top_pickup_locations_:
            feature_names.append(f"pu_loc_{loc}")
        for loc in self.top_dropoff_locations_:
            feature_names.append(f"do_loc_{loc}")

        self.feature_names_ = feature_names
        return self

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform batch DataFrame into ML feature matrix.

        Args:
            df: DataFrame containing raw trip columns (tpep_pickup_datetime, PULocationID, etc.).

        Returns:
            Processed DataFrame with extracted features matching self.feature_names_.
        """
        res = pd.DataFrame(index=df.index)

        # Standard numeric features
        res["trip_distance"] = df["trip_distance"].astype(np.float32)
        res["passenger_count"] = df["passenger_count"].astype(np.int32)

        # Datetime features
        pickup_dt = pd.to_datetime(df["tpep_pickup_datetime"])
        res["pickup_hour"] = pickup_dt.dt.hour.astype(np.int32)
        res["pickup_dayofweek"] = pickup_dt.dt.dayofweek.astype(np.int32)
        res["is_weekend"] = (res["pickup_dayofweek"] >= 5).astype(np.int32)

        # Airport flags
        pu_ids = df["PULocationID"].astype(np.int32)
        do_ids = df["DOLocationID"].astype(np.int32)
        res["is_airport_pickup"] = pu_ids.isin(AIRPORT_LOCATION_IDS).astype(np.int32)
        res["is_airport_dropoff"] = do_ids.isin(AIRPORT_LOCATION_IDS).astype(np.int32)

        # One-hot encoding for learned top locations
        for loc in self.top_pickup_locations_:
            res[f"pu_loc_{loc}"] = (pu_ids == loc).astype(np.int32)

        for loc in self.top_dropoff_locations_:
            res[f"do_loc_{loc}"] = (do_ids == loc).astype(np.int32)

        return res[self.feature_names_]

    def transform_single(self, trip: TripInput) -> np.ndarray:
        """Transform a single online inference request into a 2D float32 numpy array.

        Executed on the critical path of FastAPI serving; optimized for sub-millisecond latency.

        Args:
            trip: Validated Pydantic TripInput instance.

        Returns:
            2D numpy array of shape (1, n_features) with dtype float32.
        """
        pickup_dt = pd.to_datetime(trip.pickup_datetime)
        pickup_hour = pickup_dt.hour
        pickup_dayofweek = pickup_dt.dayofweek
        is_weekend = 1.0 if pickup_dayofweek >= 5 else 0.0

        is_airport_pu = 1.0 if trip.pickup_location_id in AIRPORT_LOCATION_IDS else 0.0
        is_airport_do = 1.0 if trip.dropoff_location_id in AIRPORT_LOCATION_IDS else 0.0

        features: list[float] = [
            float(trip.trip_distance),
            float(trip.passenger_count),
            float(pickup_hour),
            float(pickup_dayofweek),
            is_weekend,
            is_airport_pu,
            is_airport_do,
        ]

        for loc in self.top_pickup_locations_:
            features.append(1.0 if trip.pickup_location_id == loc else 0.0)

        for loc in self.top_dropoff_locations_:
            features.append(1.0 if trip.dropoff_location_id == loc else 0.0)

        return np.array([features], dtype=np.float32)

    def save(self, filepath: Path) -> None:
        """Serialize transformer instance to disk using joblib.

        Args:
            filepath: Destination file path.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "MobilityFeatureTransformer":
        """Load serialized transformer from disk.

        Args:
            filepath: Path to saved transformer joblib file.

        Returns:
            Loaded MobilityFeatureTransformer instance.
        """
        return joblib.load(filepath)
