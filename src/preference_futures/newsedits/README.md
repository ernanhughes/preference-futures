# NewsEdits adapter

The importable adapter is split into small modules:

- `schema.py`: SQLite table and column discovery
- `database.py`: read-only loading and deterministic article sampling
- `text.py`: sentence splitting and normalisation
- `extract.py`: V0 → V1 preference pairs and V1 → V2 fate resolution
- `models.py`: typed records and extraction audit
- `cli.py`: inspection and JSONL extraction

The older `newsedits.py`, `newsedits_ablation.py`, and `probe.py` scripts remain as research provenance. New code should import from `preference_futures.newsedits`, not from those scripts. They can be retired after the new modules reproduce the original experiment outputs.
