"""Dataset viability and shortcut audits for preference-futures episodes."""

from preference_futures.audit.numeric import (
    NUMERIC_AUDIT_SCHEMA_VERSION,
    build_numeric_shortcut_report,
    classify_numeric_episode,
    render_numeric_shortcut_markdown,
)
from preference_futures.audit.report import (
    CONTEXT_AUDIT_SCHEMA_VERSION,
    build_context_viability_report,
    load_episode_records,
    render_context_viability_markdown,
)

__all__ = [
    "CONTEXT_AUDIT_SCHEMA_VERSION",
    "NUMERIC_AUDIT_SCHEMA_VERSION",
    "build_context_viability_report",
    "build_numeric_shortcut_report",
    "classify_numeric_episode",
    "load_episode_records",
    "render_context_viability_markdown",
    "render_numeric_shortcut_markdown",
]
