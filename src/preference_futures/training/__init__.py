"""Fixed-budget source-task training for preference-to-future transfer experiments."""

from preference_futures.training.contract import (
    build_training_contract,
    render_training_plan_markdown,
    validate_training_contract,
    write_training_contract,
)
from preference_futures.training.data import (
    ClassificationExample,
    MaskedLanguageExample,
    SourceStore,
    deterministic_training_batches,
    load_source_store,
    materialize_record,
)
from preference_futures.training.runtime import prepare_training, run_training_jobs
from preference_futures.training.verify import (
    render_training_verification_markdown,
    verify_training_runs,
    write_training_verification,
)

__all__ = [
    "ClassificationExample",
    "MaskedLanguageExample",
    "SourceStore",
    "build_training_contract",
    "deterministic_training_batches",
    "load_source_store",
    "materialize_record",
    "prepare_training",
    "render_training_plan_markdown",
    "render_training_verification_markdown",
    "run_training_jobs",
    "validate_training_contract",
    "verify_training_runs",
    "write_training_contract",
    "write_training_verification",
]
