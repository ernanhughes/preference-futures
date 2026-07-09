# NewsEdits adapter

The importable adapter is split into small modules:

- `schema.py`: discovery for full article tables and official `split_sentences`
- `database.py`: read-only loading, source inference and deterministic sampling
- `text.py`: sentence splitting and normalisation
- `extract.py`: V0 → V1 preference pairs and V1 → V2 fate resolution
- `models.py`: typed records and extraction audit
- `cli.py`: inspection and JSONL extraction

Official source downloads such as `nyt-matched-sentences.db` contain a `split_sentences` table rather than full article text. The adapter reconstructs each article version by ordering rows by `entry_id`, `version`, and `sent_idx`, then joining the `sentence` values before applying the canonical extractor.

Both formats produce the same `NewsEditsExample` and `PreferenceEpisode` contracts.
