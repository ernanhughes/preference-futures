"""Dataset viability and shortcut audits for preference-futures episodes."""

from preference_futures.audit.report import (
    CONTEXT_AUDIT_SCHEMA_VERSION,
    build_context_viability_report,
    load_episode_records,
    render_context_viability_markdown,
)

__all__ = [
    "CONTEXT_AUDIT_SCHEMA_VERSION",
    "build_context_viability_report",
    "load_episode_records",
    "render_context_viability_markdown",
]
