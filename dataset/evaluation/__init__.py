"""TWP-Gen evaluation code.

The ten-metric automatic scoring protocol, its rubrics and the aggregation helpers.
See ``evaluation/README.md`` for how to run it.
"""

try:  # imported as a package
    from .metrics import (  # noqa: F401
        DISPLAY_NAMES,
        GROUP_RUBRICS,
        METRIC_DEFINITIONS,
        METRIC_GROUPS,
        METRIC_ORDER,
    )
except ImportError:  # imported as a flat module directory
    from metrics import (  # type: ignore  # noqa: F401
        DISPLAY_NAMES,
        GROUP_RUBRICS,
        METRIC_DEFINITIONS,
        METRIC_GROUPS,
        METRIC_ORDER,
    )
