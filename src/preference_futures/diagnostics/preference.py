"""Preference-source learnability and blinded oracle diagnostics."""

from __future__ import annotations

import math
import platform
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from preference_futures.corpora.common import hash_int
from preference_futures.training.common import (
    canonical_json_sha256,
    load_json,
    load_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from preference_futures.training.contract import validate_training_contract
from preference_futures.training.data import (
    SourceStore,
    deterministic_training_batches,
    load_source_store,
    materialize_record,
)
from preference_futures.training.runtime import (
    _collate_classification,
    _evaluate,
    _instantiate_task_model,
    _linear_schedule,
    _require_training_stack,
    _resolve_device,
    _set_seed,
)

PREFERENCE_LEARNABILITY_SCHEMA_VERSION = 1
ORACLE_SAMPLE_SCHEMA_VERSION = 1
AUTHENTIC_REGIME = "authentic_preference"
WILSON_Z_95 = 1.959963984540054


def run_preference_learnability_audit(
    *,
    training_directory: Path,
    output_directory: Path,
    fold: int = 0,
    budgets: Sequence[int] = (600, 1200, 2400, 5000, 10000),
    memorization_sizes: Sequence[int] = (256, 512),
    memorization_steps: int = 5000,
    train_evaluation_size: int = 2048,
    device: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    """Run independent preference-only diagnostics from the untouched base snapshot.

    Every condition restarts from the exact Step 3 base encoder. These runs are exploratory
    and never replace or mutate the frozen confirmatory artifacts.
    """

    training = training_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    _prepare_output(output, force=force)

    contract = load_json(training / "contract.json")
    validate_training_contract(contract)
    _require_confirmatory_parent(training, contract)
    if fold < 0 or fold >= int(contract["outer_folds"]):
        raise ValueError(f"fold must be between 0 and {int(contract['outer_folds']) - 1}")

    budgets = _positive_unique_ints(budgets, label="budgets")
    memorization_sizes = _positive_unique_ints(
        memorization_sizes,
        label="memorization_sizes",
    )
    if memorization_steps < 1:
        raise ValueError("memorization_steps must be positive")
    if train_evaluation_size < 1:
        raise ValueError("train_evaluation_size must be positive")

    job = _authentic_job(contract, fold)
    train_path = Path(str(job["train"]["path"]))
    validation_path = Path(str(job["validation"]["path"]))
    if sha256_file(train_path) != str(job["train"]["sha256"]):
        raise ValueError("authentic-preference training corpus changed")
    if sha256_file(validation_path) != str(job["validation"]["sha256"]):
        raise ValueError("authentic-preference validation corpus changed")

    train_records = load_jsonl(train_path)
    validation_records = load_jsonl(validation_path)
    source_store = load_source_store(
        Path(str(contract["sources"]["episodes"]["path"])),
        Path(str(contract["sources"]["temporal_pairs"]["path"])),
    )
    surface_baselines = evaluate_surface_baselines(validation_records, source_store)

    stack = _require_training_stack()
    torch = stack["torch"]
    resolved_device = _resolve_device(torch, device)
    snapshot = Path(str(contract["model"]["base_snapshot_path"]))
    tokenizer = stack["AutoTokenizer"].from_pretrained(snapshot / "tokenizer", use_fast=True)
    base_encoder = stack["AutoModel"].from_pretrained(snapshot / "encoder")
    base_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in base_encoder.state_dict().items()
    }
    del base_encoder

    run_directory = output / "runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    seed = int(contract["seed"]) + fold * 1000 + 700_000

    for subset_size in memorization_sizes:
        subset = balanced_record_subset(
            train_records,
            size=subset_size,
            seed=seed + subset_size,
        )
        reports.append(
            _run_condition(
                stack=stack,
                contract=contract,
                tokenizer=tokenizer,
                base_state=base_state,
                source_store=source_store,
                train_records=subset,
                train_evaluation_records=subset,
                validation_records=validation_records,
                condition_name=f"memorize-{subset_size}",
                condition_kind="tiny_set_memorization",
                update_steps=memorization_steps,
                seed=seed + subset_size * 11,
                device=resolved_device,
                output_path=run_directory / f"memorize-{subset_size}.json",
            )
        )

    train_evaluation_records = balanced_record_subset(
        train_records,
        size=min(train_evaluation_size, len(train_records)),
        seed=seed + 91,
    )
    for update_steps in budgets:
        reports.append(
            _run_condition(
                stack=stack,
                contract=contract,
                tokenizer=tokenizer,
                base_state=base_state,
                source_store=source_store,
                train_records=train_records,
                train_evaluation_records=train_evaluation_records,
                validation_records=validation_records,
                condition_name=f"budget-{update_steps}",
                condition_kind="full_train_learning_curve",
                update_steps=update_steps,
                seed=seed + update_steps,
                device=resolved_device,
                output_path=run_directory / f"budget-{update_steps}.json",
            )
        )

    summary = build_preference_learnability_summary(
        contract=contract,
        fold=fold,
        reports=reports,
        surface_baselines=surface_baselines,
    )
    write_json(output / "preference-learnability-summary.json", summary)
    (output / "preference-learnability-summary.md").write_text(
        render_preference_learnability_markdown(summary),
        encoding="utf-8",
    )
    return summary


def export_preference_oracle_sample(
    *,
    training_directory: Path,
    output_directory: Path,
    fold: int = 0,
    sample_size: int = 300,
    seed: int = 17,
    force: bool = False,
) -> dict[str, Any]:
    """Export blinded prompts and a separate answer key from one validation fold."""

    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    training = training_directory.expanduser().resolve()
    output = output_directory.expanduser().resolve()
    _prepare_output(output, force=force)
    contract = load_json(training / "contract.json")
    validate_training_contract(contract)
    job = _authentic_job(contract, fold)
    validation_records = load_jsonl(Path(str(job["validation"]["path"])))
    store = load_source_store(
        Path(str(contract["sources"]["episodes"]["path"])),
        Path(str(contract["sources"]["temporal_pairs"]["path"])),
    )
    selected = balanced_record_subset(
        validation_records,
        size=min(sample_size, len(validation_records)),
        seed=seed,
    )

    prompts: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        source_id = str(record["source_id"])
        episode = store.episodes[source_id]
        prompts.append(
            {
                "oracle_sample_schema_version": ORACLE_SAMPLE_SCHEMA_VERSION,
                "item_id": index,
                "episode_id": source_id,
                "instruction": (
                    "One sentence was retained by an editor and the other was replaced. "
                    "Using only the supplied local context, predict whether Candidate A or "
                    "Candidate B was retained. Return only A or B."
                ),
                "context_before": str(episode.get("context_before", "")),
                "candidate_a": str(episode["candidate_a"]),
                "candidate_b": str(episode["candidate_b"]),
                "context_after": str(episode.get("context_after", "")),
            }
        )
        answers.append(
            {
                "oracle_sample_schema_version": ORACLE_SAMPLE_SCHEMA_VERSION,
                "item_id": index,
                "episode_id": source_id,
                "selected": "A" if int(record["target"]) == 0 else "B",
                "selected_index": int(record["target"]),
            }
        )

    write_jsonl(output / "oracle-prompts.jsonl", prompts)
    write_jsonl(output / "oracle-answer-key.jsonl", answers)
    manifest = {
        "oracle_sample_schema_version": ORACLE_SAMPLE_SCHEMA_VERSION,
        "status": "complete",
        "fold": fold,
        "sample_size": len(selected),
        "seed": seed,
        "prompts_path": str(output / "oracle-prompts.jsonl"),
        "answer_key_path": str(output / "oracle-answer-key.jsonl"),
        "validation_source_sha256": job["validation"]["sha256"],
        "candidate_order_already_randomized": True,
        "future_labels_included": False,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    write_json(output / "oracle-manifest.json", manifest)
    return manifest


def evaluate_surface_baselines(
    records: Sequence[Mapping[str, Any]],
    store: SourceStore,
) -> dict[str, Any]:
    """Evaluate fixed, non-trained shortcuts on the authentic validation task."""

    correct = Counter()
    ties = Counter()
    total = len(records)
    for record in records:
        target = int(record["target"])
        episode = store.episodes[str(record["source_id"])]
        candidate_a = str(episode["candidate_a"])
        candidate_b = str(episode["candidate_b"])
        context = " ".join(
            (
                str(episode.get("context_before", "")),
                str(episode.get("context_after", "")),
            )
        )
        predictions = {
            "longer_candidate": _compare_scores(len(candidate_a), len(candidate_b)),
            "shorter_candidate": _compare_scores(-len(candidate_a), -len(candidate_b)),
            "context_token_overlap": _compare_scores(
                _token_overlap(candidate_a, context),
                _token_overlap(candidate_b, context),
            ),
            "more_numeric_tokens": _compare_scores(
                _numeric_token_count(candidate_a),
                _numeric_token_count(candidate_b),
            ),
        }
        for name, prediction in predictions.items():
            if prediction is None:
                ties[name] += 1
                prediction = hash_int(f"surface-tie:{name}:{record['source_id']}") % 2
            correct[name] += int(prediction == target)

    return {
        name: {
            "records": total,
            "correct": correct[name],
            "accuracy": correct[name] / max(1, total),
            "ties_resolved_deterministically": ties[name],
        }
        for name in sorted(correct)
    }


def balanced_record_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, approximately class-balanced subset."""

    if size < 1 or size > len(records):
        raise ValueError("subset size must be between 1 and the record count")
    groups = {
        label: sorted(
            (dict(record) for record in records if int(record["target"]) == label),
            key=lambda record: hash_int(
                f"diagnostic-subset:{seed}:{record['source_id']}"
            ),
        )
        for label in (0, 1)
    }
    desired = {0: size // 2, 1: size - size // 2}
    if len(groups[0]) < desired[0] or len(groups[1]) < desired[1]:
        raise ValueError("not enough records to construct the balanced subset")
    subset = groups[0][: desired[0]] + groups[1][: desired[1]]
    return sorted(subset, key=lambda record: str(record["source_id"]))


def build_preference_learnability_summary(
    *,
    contract: Mapping[str, Any],
    fold: int,
    reports: Sequence[Mapping[str, Any]],
    surface_baselines: Mapping[str, Any],
) -> dict[str, Any]:
    memorization = [
        report for report in reports if report["condition_kind"] == "tiny_set_memorization"
    ]
    learning_curve = [
        report for report in reports if report["condition_kind"] == "full_train_learning_curve"
    ]
    memorization_passed = bool(memorization) and all(
        float(report["train_evaluation"]["accuracy"]) >= 0.95
        for report in memorization
    )
    learned_runs = [
        report
        for report in learning_curve
        if report["validation_diagnostic"]["source_task_status"]
        == "learned_above_prior"
    ]
    best = min(
        learning_curve,
        key=lambda report: float(report["validation"]["mean_loss"]),
    )
    if not memorization_passed:
        outcome = "tiny_set_memorization_failed"
    elif learned_runs:
        outcome = "preference_generalised_above_prior_at_extended_budget"
    else:
        outcome = "memorisation_succeeded_but_validation_remained_null_like"

    summary: dict[str, Any] = {
        "preference_learnability_schema_version": PREFERENCE_LEARNABILITY_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "parent_contract_sha256": contract["contract_sha256"],
        "fold": fold,
        "outcome": outcome,
        "gates": {
            "tiny_set_memorization_passed": memorization_passed,
            "validation_learned_above_prior": bool(learned_runs),
        },
        "best_validation_run": {
            "condition_name": best["condition_name"],
            "update_steps": best["update_steps"],
            "accuracy": best["validation"]["accuracy"],
            "mean_loss": best["validation"]["mean_loss"],
            "source_task_status": best["validation_diagnostic"]["source_task_status"],
            "accuracy_interval_95": best["validation_diagnostic"]["accuracy_interval_95"],
        },
        "surface_baselines": dict(surface_baselines),
        "runs": list(reports),
        "interpretation_rules": {
            "tiny_set_failure": "implementation or optimisation failure remains possible",
            "memorize_but_no_generalize": (
                "the model can fit labels but the local pair-and-context task did not "
                "generalise on this fold"
            ),
            "extended_budget_success": (
                "the original 600-update budget was insufficient on this fold"
            ),
            "fresh_data_required_for_confirmation": True,
        },
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    return summary


def render_preference_learnability_markdown(summary: Mapping[str, Any]) -> str:
    best = summary["best_validation_run"]
    lines = [
        "# Step 7A Preference Learnability Audit",
        "",
        f"**Status:** {str(summary['outcome']).replace('_', ' ').upper()}",
        "",
        "This is an exploratory diagnostic performed after the frozen Steps 1-6 result.",
        "",
        "## Gates",
        "",
        f"- Tiny-set memorization passed: `{summary['gates']['tiny_set_memorization_passed']}`",
        f"- Validation learned above prior: `{summary['gates']['validation_learned_above_prior']}`",
        "",
        "## Best extended-budget validation run",
        "",
        f"- Condition: `{best['condition_name']}`",
        f"- Updates: `{best['update_steps']}`",
        f"- Accuracy: `{float(best['accuracy']):.6f}`",
        f"- Mean loss: `{float(best['mean_loss']):.6f}`",
        f"- Diagnostic: `{best['source_task_status']}`",
        (
            "- Wilson 95% interval: "
            f"`[{float(best['accuracy_interval_95'][0]):.6f}, "
            f"{float(best['accuracy_interval_95'][1]):.6f}]`"
        ),
        "",
        "## Runs",
        "",
        "| Condition | Kind | Updates | Train accuracy | Validation accuracy | Validation loss | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['condition_name']} | {run['condition_kind']} | {run['update_steps']} | "
            f"{float(run['train_evaluation']['accuracy']):.6f} | "
            f"{float(run['validation']['accuracy']):.6f} | "
            f"{float(run['validation']['mean_loss']):.6f} | "
            f"{run['validation_diagnostic']['source_task_status']} |"
        )
    lines.extend(["", "## Fixed surface baselines", ""])
    lines.append("| Baseline | Accuracy | Ties |")
    lines.append("|---|---:|---:|")
    for name, result in summary["surface_baselines"].items():
        lines.append(
            f"| {name} | {float(result['accuracy']):.6f} | "
            f"{int(result['ties_resolved_deterministically'])} |"
        )
    lines.extend(
        [
            "",
            "These runs diagnose whether the preference source task can be fitted and",
            "generalised. They do not alter the frozen confirmatory result.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_condition(
    *,
    stack: Mapping[str, Any],
    contract: Mapping[str, Any],
    tokenizer: Any,
    base_state: Mapping[str, Any],
    source_store: SourceStore,
    train_records: Sequence[dict[str, Any]],
    train_evaluation_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    condition_name: str,
    condition_kind: str,
    update_steps: int,
    seed: int,
    device: Any,
    output_path: Path,
) -> dict[str, Any]:
    torch = stack["torch"]
    _set_seed(torch, seed)
    model = _instantiate_task_model(stack, AUTHENTIC_REGIME, base_state, contract)
    model.to(device)
    model.train()

    optimisation = contract["optimisation"]
    batch_size = int(optimisation["batch_size"])
    max_length = int(optimisation["maximum_sequence_length"])
    warmup_steps = min(int(optimisation["warmup_steps"]), max(0, update_steps - 1))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimisation["learning_rate"]),
        weight_decay=float(optimisation["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _linear_schedule(warmup_steps=warmup_steps, total_steps=update_steps),
    )
    batches = deterministic_training_batches(
        len(train_records),
        batch_size=batch_size,
        update_steps=update_steps,
        seed=seed,
    )
    log_every = max(1, int(optimisation["log_every_steps"]))
    trajectory: list[dict[str, Any]] = []
    running_loss = 0.0
    running_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, indices in enumerate(batches, start=1):
        examples = [
            materialize_record(train_records[index], source_store) for index in indices
        ]
        batch = _collate_classification(
            stack,
            tokenizer,
            examples,
            max_length=max_length,
            device=device,
        )
        outputs = model(**batch)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss in {condition_name} at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(optimisation["gradient_clip_norm"]),
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += float(loss.detach().cpu())
        running_steps += 1
        if step % log_every == 0 or step == update_steps:
            trajectory.append(
                {
                    "step": step,
                    "mean_training_loss_since_last_log": running_loss / running_steps,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "clipped_gradient_norm": float(gradient_norm.detach().cpu()),
                    "examples_seen": step * batch_size,
                }
            )
            running_loss = 0.0
            running_steps = 0

    train_evaluation = _evaluate(
        stack=stack,
        model=model,
        tokenizer=tokenizer,
        records=list(train_evaluation_records),
        source_store=source_store,
        regime=AUTHENTIC_REGIME,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    validation = _evaluate(
        stack=stack,
        model=model,
        tokenizer=tokenizer,
        records=list(validation_records),
        source_store=source_store,
        regime=AUTHENTIC_REGIME,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    validation_diagnostic = classification_diagnostic(validation, validation_records)
    movement = _encoder_movement(model.base_model.state_dict(), base_state)
    report: dict[str, Any] = {
        "preference_learnability_schema_version": PREFERENCE_LEARNABILITY_SCHEMA_VERSION,
        "status": "complete",
        "exploratory": True,
        "condition_name": condition_name,
        "condition_kind": condition_kind,
        "seed": seed,
        "update_steps": update_steps,
        "training_records": len(train_records),
        "train_evaluation_records": len(train_evaluation_records),
        "validation_records": len(validation_records),
        "train_evaluation": train_evaluation,
        "validation": validation,
        "validation_diagnostic": validation_diagnostic,
        "encoder_movement": movement,
        "trajectory": trajectory,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": stack["transformers_version"],
            "device": str(device),
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json(output_path, report)
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def classification_diagnostic(
    validation: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    targets = [int(record["target"]) for record in records]
    counts = Counter(targets)
    total = len(targets)
    accuracy = float(validation["accuracy"])
    correct = int(round(accuracy * total))
    positive_rate = counts[1] / max(1, total)
    prior_accuracy = max(positive_rate, 1.0 - positive_rate)
    prior_log_loss = _binary_entropy(positive_rate)
    lower, upper = _wilson_interval(correct, total)
    mean_loss = float(validation["mean_loss"])
    if lower > prior_accuracy and mean_loss < prior_log_loss:
        status = "learned_above_prior"
    elif upper < prior_accuracy:
        status = "below_prior"
    else:
        status = "null_like"
    return {
        "records": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_interval_95": [lower, upper],
        "target_counts": {"0": counts[0], "1": counts[1]},
        "class_prior_accuracy": prior_accuracy,
        "class_prior_log_loss": prior_log_loss,
        "mean_loss": mean_loss,
        "source_task_status": status,
    }


def _encoder_movement(
    final_state: Mapping[str, Any],
    base_state: Mapping[str, Any],
) -> dict[str, float]:
    delta_sq = 0.0
    base_sq = 0.0
    for name, base in base_state.items():
        final = final_state[name].detach().cpu()
        delta = final.float() - base.float()
        delta_sq += float(delta.square().sum().item())
        base_sq += float(base.float().square().sum().item())
    delta_norm = math.sqrt(delta_sq)
    base_norm = math.sqrt(base_sq)
    return {
        "encoder_l2_delta": delta_norm,
        "base_encoder_l2_norm": base_norm,
        "relative_l2_delta": delta_norm / base_norm if base_norm else 0.0,
    }


def _authentic_job(contract: Mapping[str, Any], fold: int) -> Mapping[str, Any]:
    for job in contract["jobs"]:
        if int(job["fold"]) == fold and str(job["regime"]) == AUTHENTIC_REGIME:
            return job
    raise ValueError(f"no authentic-preference job exists for fold {fold}")


def _require_confirmatory_parent(
    training_directory: Path,
    contract: Mapping[str, Any],
) -> None:
    verification = load_json(
        training_directory / "training-verification-confirmatory.json"
    )
    if verification.get("passed") is not True or verification.get("mode") != "confirmatory":
        raise ValueError("the parent Step 3 confirmatory verification has not passed")
    observed = verification.get("observed", {})
    expected = int(contract["expected_training_jobs"])
    if observed.get("observed_jobs") != expected:
        raise ValueError("the parent Step 3 job count is incomplete")


def _prepare_output(output: Path, *, force: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise ValueError(f"output directory is not empty; pass --force: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def _positive_unique_ints(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed or any(value < 1 for value in parsed):
        raise ValueError(f"{label} must contain positive integers")
    return parsed


def _compare_scores(left: float, right: float) -> int | None:
    if left > right:
        return 0
    if right > left:
        return 1
    return None


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in value.split() if token.strip()}


def _token_overlap(candidate: str, context: str) -> float:
    candidate_tokens = _tokens(candidate)
    context_tokens = _tokens(context)
    if not candidate_tokens or not context_tokens:
        return 0.0
    return len(candidate_tokens & context_tokens) / len(candidate_tokens | context_tokens)


def _numeric_token_count(value: str) -> int:
    return sum(any(character.isdigit() for character in token) for token in value.split())


def _binary_entropy(rate: float) -> float:
    if rate <= 0.0 or rate >= 1.0:
        return 0.0
    return -(rate * math.log(rate) + (1.0 - rate) * math.log(1.0 - rate))


def _wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total < 1:
        raise ValueError("Wilson interval requires at least one observation")
    rate = correct / total
    z2 = WILSON_Z_95**2
    denominator = 1.0 + z2 / total
    centre = (rate + z2 / (2.0 * total)) / denominator
    spread = (
        WILSON_Z_95
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)
