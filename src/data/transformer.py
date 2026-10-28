"""Feature engineering and preprocessing pipeline for fraud detection.

Backward compatibility re-exports from src.domains.fraud.transformer.
"""

from src.domains.fraud.transformer import (
    DEFAULT_IP_MAPPING_PATH,
    TOP_COUNTRIES_DEFAULT,
    FraudFeatureTransformer,
    IpCountryLookup,
)

__all__ = [
    "DEFAULT_IP_MAPPING_PATH",
    "TOP_COUNTRIES_DEFAULT",
    "IpCountryLookup",
    "FraudFeatureTransformer",
]
