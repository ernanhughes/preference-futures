"""Compute-matched source-task corpora for preference-transfer experiments."""

from preference_futures.corpora.build import (
    CORPUS_NAMES,
    build_compute_matched_corpora,
    render_corpus_summary_markdown,
    write_compute_matched_corpora,
)
from preference_futures.corpora.temporal import extract_independent_temporal_pairs

__all__ = [
    "CORPUS_NAMES",
    "build_compute_matched_corpora",
    "extract_independent_temporal_pairs",
    "render_corpus_summary_markdown",
    "write_compute_matched_corpora",
]
