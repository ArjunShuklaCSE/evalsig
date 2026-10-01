"""evalsig: is the new version of your AI system really better, or is it noise?"""

from .core import (
    Comparison,
    EvalsigError,
    MetricComparison,
    MetricSummary,
    Summary,
    compare_arrays,
    summarize_arrays,
)

__all__ = [
    "Comparison",
    "EvalsigError",
    "MetricComparison",
    "MetricSummary",
    "Summary",
    "compare_arrays",
    "summarize_arrays",
]
