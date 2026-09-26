"""An auditable no-ML baseline used only for comparison, never silently selected."""
from __future__ import annotations

def is_match(features: dict[str, float]) -> bool:
    # Conservative by design: makes the precision/recall tradeoff visible in experiments.
    return bool(features["country_equal"] and ((features["name_exact"] and features["address_ratio"] >= .55) or (features["address_exact"] and features["name_ratio"] >= .72)))
