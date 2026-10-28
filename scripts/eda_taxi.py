"""Exploratory Data Analysis (EDA) on NYC Yellow Taxi Trip Records (2019 vs 2020).

Used to inspect real-world covariate and concept drift caused by the March 2020 COVID-19 lockdown,
evaluating baseline model behavior and selecting the optimal target for the MLOps runtime.
"""

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_DIR = Path("data/raw/taxi")


def clean_taxi_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw taxi trips to remove obvious recording errors and outliers.

    Args:
        df: Raw taxi trip DataFrame.

    Returns:
        Cleaned DataFrame with computed duration and target columns.
    """
    df = df.copy()

    # Valid physical bounds
    df = df[
        (df["trip_distance"] > 0)
        & (df["trip_distance"] < 100)
        & (df["fare_amount"] > 2.5)
        & (df["fare_amount"] < 500)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    ]

    pickup = pd.to_datetime(df["tpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["duration_minutes"] = (dropoff - pickup).dt.total_seconds() / 60.0
    df = df[(df["duration_minutes"] >= 1.0) & (df["duration_minutes"] <= 180.0)]

    # Feature temporali
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek

    # Velocità in mph (miglia / ora)
    df["speed_mph"] = df["trip_distance"] / (df["duration_minutes"] / 60.0)

    # TARGET 1: Mancia generosa (>= 20% con carta di credito)
    df["tip_fraction"] = df["tip_amount"] / df["fare_amount"]
    df["high_tip"] = (df["tip_fraction"] >= 0.20).astype(int)

    # TARGET 2: Corsa congestionata / lenta (< 12 mph, traffico tipico pre-covid)
    df["is_congested_trip"] = (df["speed_mph"] < 12.0).astype(int)

    return df


def main() -> None:
    print("=" * 70)
    print("STEP 1: Caricamento dati campionati (2019 baseline vs 2020 lockdown)...")

    # 1. 2019-01 Baseline (campionamento uniforme di 100k su 7.6 milioni)
    print("  Caricamento 2019-01 (Baseline normale pre-covid)...")
    df_2019_raw = pd.read_parquet(DATA_DIR / "yellow_tripdata_2019-01.parquet")
    df_2019 = df_2019_raw.sample(n=100_000, random_state=42)

    # 2. 2020-03 Lockdown reale (25-31 marzo 2020, dopo il New York State on PAUSE order)
    print("  Caricamento 2020-03 (Lockdown pieno a New York)...")
    df_2020_raw = pd.read_parquet(DATA_DIR / "yellow_tripdata_2020-03.parquet")
    df_2020_raw["tpep_pickup_datetime"] = pd.to_datetime(df_2020_raw["tpep_pickup_datetime"])
    df_2020_lockdown = df_2020_raw[
        (df_2020_raw["tpep_pickup_datetime"] >= "2020-03-25")
        & (df_2020_raw["tpep_pickup_datetime"] <= "2020-03-31")
    ]

    print(f"  Righe campionate 2019: {len(df_2019):,}")
    print(f"  Righe totali lockdown (ultimi 7 gg marzo 2020): {len(df_2020_lockdown):,}")

    df_2019_clean = clean_taxi_data(df_2019)
    df_2020_clean = clean_taxi_data(df_2020_lockdown)

    print("\n" + "=" * 70)
    print("STEP 2: EVIDENZA DEL COVARIATE & BEHAVIORAL DRIFT")
    v_19 = df_2019_clean["speed_mph"].median()
    v_20 = df_2020_clean["speed_mph"].median()
    print(f"  Velocità media (mph):        2019 = {v_19:.1f} mph | Lockdown 2020 = {v_20:.1f} mph")

    d_19 = df_2019_clean["duration_minutes"].mean()
    d_20 = df_2020_clean["duration_minutes"].mean()
    print(f"  Durata media (min):          2019 = {d_19:.1f} min | Lockdown 2020 = {d_20:.1f} min")

    dist_19 = df_2019_clean["trip_distance"].mean()
    dist_20 = df_2020_clean["trip_distance"].mean()
    print(
        f"  Distanza media (miglia):     2019 = {dist_19:.2f} mi | "
        f"Lockdown 2020 = {dist_20:.2f} mi"
    )

    # Quota corse per l'aeroporto JFK (Location ID 132)
    jfk_19 = (df_2019_clean["DOLocationID"] == 132).mean()
    jfk_20 = (df_2020_clean["DOLocationID"] == 132).mean()
    jfk_drop = (1.0 - jfk_20 / jfk_19) if jfk_19 > 0 else 0.0
    print(
        f"  Quota destinazione JFK:      2019 = {jfk_19:.2%} | "
        f"Lockdown 2020 = {jfk_20:.2%} (Crollo: {jfk_drop:.1%})"
    )

    # Quota corse congestionate / a rilento (< 12 mph)
    cong_19 = df_2019_clean["is_congested_trip"].mean()
    cong_20 = df_2020_clean["is_congested_trip"].mean()
    print(f"  Tasso corse congestionate:   2019 = {cong_19:.2%} | Lockdown 2020 = {cong_20:.2%}")

    print("\n" + "=" * 70)
    print("STEP 3: TEST MODELLO PRE-DRIFT (2019) vs DRIFT COVID (2020)")
    print("Target: 'is_congested_trip' (Predizione traffico / lentezza corsa)")
    print("Features: PULocationID, DOLocationID, trip_distance, passenger_count, pickup_hour, dow")

    features = [
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "passenger_count",
        "pickup_hour",
        "pickup_dayofweek",
    ]

    split_idx = int(len(df_2019_clean) * 0.8)
    train_df = df_2019_clean.iloc[:split_idx]
    val_df = df_2019_clean.iloc[split_idx:]

    clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf.fit(train_df[features], train_df["is_congested_trip"])

    # Valutazione
    p_val = clf.predict_proba(val_df[features])[:, 1]
    p_2020 = clf.predict_proba(df_2020_clean[features])[:, 1]

    auc_val = roc_auc_score(val_df["is_congested_trip"], p_val)
    auc_2020 = roc_auc_score(df_2020_clean["is_congested_trip"], p_2020)

    print(f"\n  [2019 Validation]  ROC-AUC: {auc_val:.4f}  (Ottimo segnale pre-drift)")
    print(f"  [2020 Lockdown]    ROC-AUC: {auc_2020:.4f}  (DEGRADO DA DRIFT)")
    print(f"  Degrado relativo ROC-AUC: {(auc_val - auc_2020) / auc_val:.2%}")

    print("\nReport su dati 2020 con soglia 0.5:")
    report = classification_report(
        df_2020_clean["is_congested_trip"],
        (p_2020 >= 0.5).astype(int),
        target_names=["Fluida", "Congestionata"],
    )
    print(report)
    print("=" * 70)


if __name__ == "__main__":
    main()
