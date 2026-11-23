"""Which harness checks run, which conditions use them, and the settings recorded in the run manifest."""
from __future__ import annotations

from .base import Check, CheckResult
from .format_check import FormatCheck
from .freshness import FRESHNESS_DAYS, FreshnessCheck

__all__ = ["Check", "CheckResult", "HARNESS_CHECKS", "HARNESS_CONDITIONS", "LINE_PREFIX", "harness_settings"]

# Conditions whose tool responses go through the harness layer
HARNESS_CONDITIONS = ("harness", "flags_only")

LINE_PREFIX = "_harness: "

# The checks the harness runs, in the order they appear in the _harness line
HARNESS_CHECKS: list[Check] = [FormatCheck(), FreshnessCheck()]


def harness_settings(condition: str) -> dict | None:
    """The harness settings recorded in the run manifest; None for conditions without the harness."""
    if condition not in HARNESS_CONDITIONS:
        return None
    return {
        "checks": [check.key for check in HARNESS_CHECKS],
        "freshness_days": FRESHNESS_DAYS,
        "line_prefix": LINE_PREFIX,
    }
