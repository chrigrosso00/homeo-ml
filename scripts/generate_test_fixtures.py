"""Synthetic test fixtures and baseline artifacts generator for CI/CD environments.

Ensures that clean CI checkouts (where large raw datasets and binary models are
ignored by git) have valid, minimal, lightweight test fixtures and artifacts
to execute the full unit and integration test suite and container builds.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fixtures")

DATA_FRAUD_DIR = Path("data/raw/fraud")
DATA_TAXI_DIR = Path("data/raw/taxi")
ARTIFACTS_MODELS_DIR = Path("artifacts/models")
ARTIFACTS_MOBILITY_DIR = Path("artifacts/mobility")
ARTIFACTS_CANARY_DIR = Path("artifacts/canary")


def ensure_fraud_raw_fixtures() -> None:
    """Generate minimal synthetic CSV fixtures for fraud detection if missing."""
    DATA_FRAUD_DIR.mkdir(parents=True, exist_ok=True)

    ip_csv = DATA_FRAUD_DIR / "IpAddress_to_Country.csv"
    if not ip_csv.exists():
        logger.info("Generating synthetic IP-to-Country mapping: %s", ip_csv)
        df_ip = pd.DataFrame(
            {
                "lower_bound_ip_address": [0.0, 350311000.0, 732758000.0],
                "upper_bound_ip_address": [100.0, 350312000.0, 732759000.0],
                "country": ["Unknown", "United States", "Japan"],
            }
        )
        df_ip.to_csv(ip_csv, index=False)

    fraud_csv = DATA_FRAUD_DIR / "Fraud_Data.csv"
    if not fraud_csv.exists():
        logger.info("Generating synthetic Fraud_Data CSV: %s", fraud_csv)
        n = 500
        records = []
        for i in range(n):
            day = 1 + (i // 20)
            is_fraud = 1 if (i % 6 == 0) else 0
            signup = f"2020-01-{day:02d} 08:00:00"
            purchase = f"2020-01-{day:02d} 08:00:02" if is_fraud else f"2020-01-{day:02d} 14:00:00"
            records.append(
                {
                    "user_id": 1000 + i,
                    "signup_time": signup,
                    "purchase_time": purchase,
                    "purchase_value": 150.0 if is_fraud else 45.0,
                    "device_id": f"DEV_{i % 20}",
                    "source": "Ads" if i % 2 == 0 else "SEO",
                    "browser": "Chrome" if i % 3 == 0 else "Safari",
                    "sex": "M" if i % 2 == 0 else "F",
                    "age": 25 + (i % 40),
                    "ip_address": 732758400.0 if i % 2 == 0 else 350311400.0,
                    "class": is_fraud,
                }
            )
        df_fraud = pd.DataFrame(records)
        df_fraud.to_csv(fraud_csv, index=False)
        # Also ensure legacy flat path exists for backward compatibility
        legacy_csv = Path("data/raw/Fraud_Data.csv")
        if not legacy_csv.exists():
            df_fraud.to_csv(legacy_csv, index=False)


def ensure_taxi_raw_fixtures() -> None:
    """Generate minimal synthetic parquet fixtures for NYC Taxi if missing."""
    DATA_TAXI_DIR.mkdir(parents=True, exist_ok=True)

    months = ["2019-01", "2019-02", "2019-03", "2020-01", "2020-02", "2020-03"]
    for m in months:
        parquet_file = DATA_TAXI_DIR / f"yellow_tripdata_{m}.parquet"
        if not parquet_file.exists():
            logger.info("Generating synthetic taxi parquet: %s", parquet_file)
            n = 500
            records = []
            is_lockdown = m == "2020-03"
            for i in range(n):
                day = 26 if (is_lockdown and i > 200) else (1 + (i % 25))
                pickup = f"{m}-{day:02d} 14:00:00"
                # Vary duration to have both congested (slow) and free-flow (fast) trips
                duration_m = 25 if (i % 2 == 0) else 8
                dropoff = f"{m}-{day:02d} 14:{duration_m:02d}:00"
                records.append(
                    {
                        "VendorID": 1,
                        "tpep_pickup_datetime": pickup,
                        "tpep_dropoff_datetime": dropoff,
                        "passenger_count": 1,
                        "trip_distance": 2.5,
                        "RatecodeID": 1,
                        "store_and_fwd_flag": "N",
                        "PULocationID": 161,
                        "DOLocationID": 236,
                        "payment_type": 1,
                        "fare_amount": 15.0,
                        "extra": 1.0,
                        "mta_tax": 0.5,
                        "tip_amount": 3.0,
                        "tolls_amount": 0.0,
                        "improvement_surcharge": 0.3,
                        "total_amount": 19.8,
                        "congestion_surcharge": 2.5,
                        "airport_fee": 0.0,
                    }
                )
            pd.DataFrame(records).to_parquet(parquet_file, index=False)


def ensure_model_artifacts() -> None:
    """Train minimal baseline models and export ONNX if artifacts are absent."""
    fraud_onnx = ARTIFACTS_MODELS_DIR / "fraud_model.onnx"
    if not fraud_onnx.exists():
        logger.info("Training minimal baseline fraud model for CI artifacts...")
        from src.models.train import train_baseline_pipeline

        train_baseline_pipeline(
            data_path=DATA_FRAUD_DIR / "Fraud_Data.csv",
            artifacts_dir=ARTIFACTS_MODELS_DIR,
            model_version="v1.0.0",
        )

    mobility_onnx = ARTIFACTS_MOBILITY_DIR / "mobility_model.onnx"
    if not mobility_onnx.exists():
        logger.info("Training minimal baseline mobility model for CI artifacts...")
        from src.models.train_mobility import train_mobility_pipeline

        train_mobility_pipeline(
            data_dir=DATA_TAXI_DIR,
            artifacts_dir=ARTIFACTS_MOBILITY_DIR,
            model_version="v1.0.0",
            samples_per_month=200,
        )

    canary_onnx = ARTIFACTS_CANARY_DIR / "fraud_model.onnx"
    if not canary_onnx.exists():
        logger.info("Initializing canary artifact directory: %s", ARTIFACTS_CANARY_DIR)
        ARTIFACTS_CANARY_DIR.mkdir(parents=True, exist_ok=True)
        for f in ARTIFACTS_MODELS_DIR.iterdir():
            if f.is_file():
                shutil.copy(f, ARTIFACTS_CANARY_DIR / f.name)


def main() -> None:
    """Generate all missing test fixtures and baseline artifacts."""
    logger.info("Ensuring test fixtures and baseline artifacts are present...")
    ensure_fraud_raw_fixtures()
    ensure_taxi_raw_fixtures()
    ensure_model_artifacts()
    logger.info("All test fixtures and baseline artifacts verified successfully.")


if __name__ == "__main__":
    main()
