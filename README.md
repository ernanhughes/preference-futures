# preference-futures

Research code for testing whether preference-specific learning transfers to prediction of a later, decision-linked outcome.

## Episode contract

A revision triplet is represented as:

```text
V0: rejected sentence
V1: selected replacement
V2: later state of the selected replacement
```

The episode builder randomises whether V1 appears as candidate A or B, while preserving both the preference label and the future outcome attached to V1.

Every exported episode is a versioned, JSON-compatible record containing:

```text
schema_version
episode_id
lineage_id
candidate_a
candidate_b
selected_index
future_revised
```

Whitespace-only differences between V1 and V2 are treated as stable. V0 and V1 must remain textually distinct after whitespace normalisation.

## Repository sequence

```text
1. Canonical episode contract
2. NewsEdits source adapter
3. Context viability audit
4. Grouped split manifests
5. Preference-task baselines
6. Frozen representation transfer
7. Sample-efficiency and shortcut controls
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```
