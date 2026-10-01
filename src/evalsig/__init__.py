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
from .data import compare, load_rows, summarize

__version__ = "0.1.0"

__all__ = [
    "Comparison",
    "EvalsigError",
    "MetricComparison",
    "MetricSummary",
    "Summary",
    "compare",
    "compare_arrays",
    "load_rows",
    "summarize",
    "summarize_arrays",
]
