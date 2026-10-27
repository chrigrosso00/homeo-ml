"""Traffic generator and controlled drift injection simulator.

Simulates production payment gateway traffic by streaming transactions to the
FastAPI serving layer. Supports configurable scenarios to trigger covariate shift
and concept drift in real time.
"""

import argparse
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_ENDPOINT: str = "http://localhost:8000/predict"
DEFAULT_DATA_PATH: Path = Path("data/raw/Fraud_Data.csv")


def generate_normal_transaction(row: pd.Series) -> dict[str, Any]:
    """Format an existing historical test transaction into a valid API payload.

    Args:
        row: Pandas Series representing a real transaction row.

    Returns:
        JSON-serializable dictionary matching TransactionInput schema.
    """
    signup = pd.to_datetime(row["signup_time"])
    purchase = pd.to_datetime(row["purchase_time"])

    return {
        "user_id": int(row["user_id"]),
        "signup_time": signup.isoformat(),
        "purchase_time": purchase.isoformat(),
        "purchase_value": float(row["purchase_value"]),
        "device_id": str(row["device_id"]),
        "source": str(row["source"]),
        "browser": str(row["browser"]),
        "sex": str(row["sex"]),
        "age": int(row["age"]),
        "ip_address": float(row["ip_address"]),
    }


def generate_botnet_drift_transaction(user_idx: int) -> dict[str, Any]:
    """Generate an aggressive synthetic botnet attack transaction.

    Simulates instant checkout (<2 seconds from registration), browser spoofing,
    and heavy device sharing across separate accounts.

    Args:
        user_idx: Counter for unique customer ID.

    Returns:
        Perturbed TransactionInput payload.
    """
    now = datetime.now(UTC)
    delta_seconds = random.uniform(0.1, 2.0)
    signup_time = now - timedelta(seconds=delta_seconds)

    # 5 shared bot device pools
    shared_device = f"BOT_DEV_POOL_{random.randint(1, 5)}"

    return {
        "user_id": 900000 + user_idx,
        "signup_time": signup_time.isoformat(),
        "purchase_time": now.isoformat(),
        "purchase_value": round(random.uniform(15.0, 85.0), 2),
        "device_id": shared_device,
        "source": "Direct",
        "browser": random.choice(["Opera", "Safari"]),
        "sex": random.choice(["M", "F"]),
        "age": random.randint(22, 38),
        "ip_address": 2621474000.0 + float(random.randint(1, 100)),
    }


def generate_holiday_drift_transaction(user_idx: int) -> dict[str, Any]:
    """Generate a holiday shopping surge with inflated purchase values.

    Args:
        user_idx: Counter for unique customer ID.

    Returns:
        Perturbed TransactionInput payload with high value amounts.
    """
    now = datetime.now(UTC)
    signup_time = now - timedelta(days=random.randint(10, 180))

    return {
        "user_id": 800000 + user_idx,
        "signup_time": signup_time.isoformat(),
        "purchase_time": now.isoformat(),
        "purchase_value": round(random.uniform(450.0, 1500.0), 2),  # Inflated amounts
        "device_id": f"HOLIDAY_USER_{user_idx}",
        "source": "Ads",
        "browser": "Chrome",
        "sex": random.choice(["M", "F"]),
        "age": random.randint(25, 60),
        "ip_address": 732758400.0 + float(random.randint(1, 200)),
    }


def run_traffic_simulation(
    endpoint: str = DEFAULT_ENDPOINT,
    scenario: str = "normal",
    count: int = 250,
    delay_sec: float = 0.01,
    data_path: Path = DEFAULT_DATA_PATH,
) -> None:
    """Execute live HTTP transaction stream against FastAPI serving layer.

    Args:
        endpoint: Destination /predict URL.
        scenario: Traffic pattern ('normal', 'botnet', 'holiday').
        count: Number of transactions to dispatch.
        delay_sec: Inter-request sleep interval in seconds.
        data_path: Path to raw dataset CSV for normal scenario replay.
    """
    print("\n=======================================================")
    print(f"Starting Traffic Simulator: Scenario='{scenario.upper()}'")
    print(f"Target: {endpoint} | Total Requests: {count} | Delay: {delay_sec}s")
    print("=======================================================\n")

    test_pool: list[pd.Series] = []
    if scenario == "normal":
        df = pd.read_csv(data_path)
        # Take the latter test partition
        test_df = df.iloc[int(len(df) * 0.8) :].reset_index(drop=True)
        test_pool = [test_df.iloc[i] for i in range(min(count * 2, len(test_df)))]

    latencies: list[float] = []
    fraud_flagged = 0
    success_count = 0

    session = requests.Session()

    for i in range(1, count + 1):
        if scenario == "normal":
            row = test_pool[i % len(test_pool)]
            payload = generate_normal_transaction(row)
        elif scenario == "botnet":
            payload = generate_botnet_drift_transaction(i)
        elif scenario == "holiday":
            payload = generate_holiday_drift_transaction(i)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        t0 = time.perf_counter()
        try:
            resp = session.post(endpoint, json=payload, timeout=5.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            if resp.status_code == 200:
                success_count += 1
                res_json = resp.json()
                if res_json.get("is_fraud"):
                    fraud_flagged += 1

                if i % 25 == 0 or i == count:
                    flag_rate = (fraud_flagged / success_count) * 100.0
                    avg_lat = sum(latencies[-25:]) / len(latencies[-25:])
                    print(
                        f"  [Tx {i:03d}/{count}] Status=200 | Latency={avg_lat:.2f}ms | "
                        f"Fraud Flag Rate={flag_rate:.1f}% ({fraud_flagged}/{success_count})"
                    )
            else:
                print(f"  [Tx {i:03d}/{count}] Error HTTP {resp.status_code}: {resp.text[:100]}")

        except Exception as exc:
            print(f"  [Tx {i:03d}/{count}] Connection failed: {exc}")

        if delay_sec > 0:
            time.sleep(delay_sec)

    avg_total_lat = sum(latencies) / max(len(latencies), 1)
    print("\nSimulation Complete!")
    print(f"  Delivered: {success_count}/{count} requests")
    print(f"  Average HTTP Latency: {avg_total_lat:.2f}ms")
    print(f"  Frauds Detected: {fraud_flagged} ({(fraud_flagged/max(success_count,1))*100:.1f}%)")


def main() -> None:
    """CLI entry point for traffic simulator."""
    parser = argparse.ArgumentParser(description="Simulate fraud detection production traffic.")
    parser.add_argument("--url", default=DEFAULT_ENDPOINT, help="Predict endpoint URL")
    parser.add_argument(
        "--scenario",
        default="normal",
        choices=["normal", "botnet", "holiday"],
        help="Traffic scenario pattern to inject",
    )
    parser.add_argument("--count", type=int, default=250, help="Number of transactions to stream")
    parser.add_argument(
        "--delay", type=float, default=0.01, help="Delay between requests in seconds"
    )

    args = parser.parse_args()
    run_traffic_simulation(
        endpoint=args.url,
        scenario=args.scenario,
        count=args.count,
        delay_sec=args.delay,
    )


if __name__ == "__main__":
    main()
