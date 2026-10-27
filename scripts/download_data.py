"""Data acquisition script for the E-Commerce Fraud Detection dataset.

Downloads raw transactions and IP-to-country mapping tables, verifies
their integrity, and stores them in data/raw/.
"""

from pathlib import Path

import pandas as pd
import requests

RAW_DATA_DIR: Path = Path("data/raw")

URL_FRAUD_DATA: str = (
    "https://raw.githubusercontent.com/pwinslow/Fraud-Detection/master/Fraud_Data/Fraud_Data.csv"
)
URL_IP_COUNTRY: str = (
    "https://raw.githubusercontent.com/pwinslow/Fraud-Detection/master/Fraud_Data/IpAddress_to_Country.csv"
)


def download_file(url: str, destination: Path) -> None:
    """Download a remote file using streaming chunks.

    Args:
        url: Remote HTTP(S) URL to download.
        destination: Local Path where the file should be saved.
    """
    if destination.exists():
        size_bytes = destination.stat().st_size
        print(f"File already exists: {destination} ({size_bytes:,} bytes). Skipping.")
        return

    print(f"Downloading {url} -> {destination}...")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Successfully downloaded: {destination} ({destination.stat().st_size:,} bytes)")


def inspect_dataset(file_path: Path) -> None:
    """Load and print summary statistics for the fraud dataset.

    Args:
        file_path: Path to the downloaded CSV file.
    """
    print(f"\n--- Summary for {file_path.name} ---")
    df = pd.read_csv(file_path)
    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample Rows:")
    print(df.head(3))

    if "class" in df.columns:
        fraud_counts = df["class"].value_counts()
        fraud_ratio = df["class"].mean() * 100
        print(f"\nClass Distribution:\n{fraud_counts}")
        print(f"Fraud Rate: {fraud_ratio:.2f}%")

    if "signup_time" in df.columns and "purchase_time" in df.columns:
        df["signup_time"] = pd.to_datetime(df["signup_time"])
        df["purchase_time"] = pd.to_datetime(df["purchase_time"])
        print(f"Date Range (Purchase): {df['purchase_time'].min()} to {df['purchase_time'].max()}")


def main() -> None:
    """Entry point for data acquisition workflow."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    fraud_data_path: Path = RAW_DATA_DIR / "Fraud_Data.csv"
    ip_country_path: Path = RAW_DATA_DIR / "IpAddress_to_Country.csv"

    download_file(URL_FRAUD_DATA, fraud_data_path)
    download_file(URL_IP_COUNTRY, ip_country_path)

    inspect_dataset(fraud_data_path)
    inspect_dataset(ip_country_path)


if __name__ == "__main__":
    main()
