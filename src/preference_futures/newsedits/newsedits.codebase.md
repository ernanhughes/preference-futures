# Codebase Pack: newsedits

```text
ROOT: C:\Projects\preference-futures\src\preference_futures\newsedits
GENERATED_AT_UTC: 2026-07-09T10:41:17.402330+00:00
PART: 1/1
FILES_IN_PART: 3
TOTAL_LINES_IN_PART: 5469
TOTAL_BYTES_UTF8_IN_PART: 171930
MODE: configured include extensions
LINE_NUMBERS: True
MAX_FILE_KB: 400
```

## How to cite this pack in review

Use the stable file ID plus line numbers, for example:

```text
F0007 `services/replay_service.py` L0042-L0068
```

## File Index

| ID | Path | Lang | Lines | KB | SHA256 |
|---|---|---:|---:|---:|---|
| F0001 | `newsedits.py` | python | 1641 | 48.5 | `50aa21c6e064` |
| F0002 | `newsedits_ablation.py` | python | 2337 | 72.1 | `e64e0ef7c3f2` |
| F0003 | `probe.py` | python | 1491 | 47.3 | `c7bfe0cac947` |

## Directory Tree

```text
└─ newsedits.py
└─ newsedits_ablation.py
└─ probe.py
```

## Files


---

## F0001 — `newsedits.py`

```text
FILE_ID: F0001
PATH: newsedits.py
LANGUAGE: python
LINES: 1641
BYTES_UTF8: 49666
SHA256: 50aa21c6e064dfdba6cee59e5c46f7cdec2652712ce52806bea7a448abac9de2
```

```python
0001 | #!/usr/bin/env python3
0002 | r"""
0003 | PreferenceFutures — NewsEdits revision-lineage probe.
0004 | 
0005 | Research question
0006 | -----------------
0007 | 
0008 | Does a real human revision event contain incremental information about the
0009 | future of the revised sentence?
0010 | 
0011 | For three consecutive article versions V0 -> V1 -> V2:
0012 | 
0013 |     rejected sentence: sentence in V0
0014 |     retained sentence: its one-to-one replacement in V1
0015 |     future: whether the retained V1 sentence is revised or removed in V2
0016 | 
0017 | The baseline already sees the retained sentence and its V1 context:
0018 | 
0019 |     P(F | context, retained sentence, metadata)
0020 | 
0021 | The preference-informed model additionally sees what the journalist replaced:
0022 | 
0023 |     P(F | context, retained sentence, rejected sentence, edit evidence, metadata)
0024 | 
0025 | Preference Future Information:
0026 | 
0027 |     PFI = Loss(baseline) - Loss(preference-informed)
0028 | 
0029 | Positive held-out PFI means the rejected alternative carries information about
0030 | the next revision beyond the retained sentence itself.
0031 | 
0032 | This is a revealed-revision-preference experiment, not an explicit A/B-vote
0033 | experiment and not a causal estimate.
0034 | 
0035 | Expected NewsEdits source
0036 | -------------------------
0037 | 
0038 | The original NewsEdits release is described as SQLite tables. This script reads
0039 | the article-version table directly and discovers column names case-
0040 | insensitively. The expected logical fields are:
0041 | 
0042 |     SOURCE, A_ID, VERSION_ID, TEXT
0043 | 
0044 | Optional fields:
0045 | 
0046 |     CREATED, TITLE, NUM_VERSIONS
0047 | 
0048 | The script does not require the precomputed sentence_diffs table. It reconstructs
0049 | consecutive sentence revisions from full article versions, which makes the
0050 | future linkage explicit and avoids depending on tag-format details.
0051 | 
0052 | Dependencies
0053 | ------------
0054 | 
0055 |     pip install pandas numpy scikit-learn
0056 | 
0057 | Optional, only for Parquet episode caches:
0058 | 
0059 |     pip install pyarrow
0060 | 
0061 | Examples
0062 | --------
0063 | 
0064 | Inspect the database schema:
0065 | 
0066 |     python preference_futures_newsedits.py \
0067 |       --db /path/to/newsedits.db \
0068 |       --inspect-only
0069 | 
0070 | Windows PowerShell smoke test:
0071 | 
0072 |     python preference_futures_newsedits.py `
0073 |       --db C:\data\newsedits.db `
0074 |       --max-articles 5000 `
0075 |       --max-episodes 50000 `
0076 |       --seeds 1,2,3 `
0077 |       --bootstrap-samples 500 `
0078 |       --episode-cache newsedits_smoke_episodes.csv.gz `
0079 |       --out newsedits_smoke_runs.csv `
0080 |       --summary-out newsedits_smoke_summary.csv
0081 | 
0082 | Larger run:
0083 | 
0084 |     python preference_futures_newsedits.py `
0085 |       --db C:\data\newsedits.db `
0086 |       --max-articles 100000 `
0087 |       --seeds 1,2,3,4,5,6,7,8,9,10 `
0088 |       --bootstrap-samples 5000 `
0089 |       --episode-cache newsedits_full_episodes.csv.gz `
0090 |       --out newsedits_full_runs.csv `
0091 |       --summary-out newsedits_full_summary.csv
0092 | 
0093 | Use --max-articles 0 to request every qualifying article. Build and validate a
0094 | smaller cache first: the complete corpus is very large.
0095 | """
0096 | 
0097 | from __future__ import annotations
0098 | 
0099 | import argparse
0100 | import dataclasses
0101 | import difflib
0102 | import hashlib
0103 | import math
0104 | import random
0105 | import re
0106 | import sqlite3
0107 | import sys
0108 | from pathlib import Path
0109 | from typing import Any, Iterable, Iterator, Sequence
0110 | 
0111 | import numpy as np
0112 | import pandas as pd
0113 | 
0114 | 
0115 | # ---------------------------------------------------------------------------
0116 | # Data records
0117 | # ---------------------------------------------------------------------------
0118 | 
0119 | 
0120 | @dataclasses.dataclass(frozen=True)
0121 | class ArticleSchema:
0122 |     table: str
0123 |     source: str
0124 |     article_id: str
0125 |     version_id: str
0126 |     text: str
0127 |     created: str | None
0128 |     title: str | None
0129 |     num_versions: str | None
0130 | 
0131 | 
0132 | @dataclasses.dataclass
0133 | class ResultRow:
0134 |     track: str
0135 |     condition: str
0136 |     seed: int
0137 |     target: str
0138 |     feature_set: str
0139 |     n_train: int
0140 |     n_test: int
0141 |     n_train_groups: int
0142 |     n_test_groups: int
0143 |     loss: float
0144 |     brier: float
0145 |     auc: float
0146 |     average_precision: float
0147 |     accuracy: float
0148 |     train_prevalence: float
0149 |     test_prevalence: float
0150 |     mean_predicted_probability: float
0151 |     null_log_loss: float
0152 |     null_brier: float
0153 | 
0154 | 
0155 | @dataclasses.dataclass
0156 | class SummaryRow:
0157 |     track: str
0158 |     condition: str
0159 |     statistic: str
0160 |     n_seeds: int
0161 |     mean: float
0162 |     seed_std: float
0163 |     ci_low: float
0164 |     ci_high: float
0165 |     positive_seeds: int
0166 |     confidence_level: float
0167 |     bootstrap_samples: int
0168 | 
0169 | 
0170 | @dataclasses.dataclass
0171 | class EvaluationBundle:
0172 |     rows: list[ResultRow]
0173 |     group_loss_stats: pd.DataFrame
0174 | 
0175 | 
0176 | # ---------------------------------------------------------------------------
0177 | # General utilities
0178 | # ---------------------------------------------------------------------------
0179 | 
0180 | 
0181 | def print_header(title: str) -> None:
0182 |     print("\n" + "=" * 100)
0183 |     print(title)
0184 |     print("=" * 100)
0185 | 
0186 | 
0187 | def parse_int_list(value: str | None, fallback: int) -> list[int]:
0188 |     if value is None or not value.strip():
0189 |         return [fallback]
0190 |     values = [int(part.strip()) for part in value.split(",") if part.strip()]
0191 |     if not values:
0192 |         raise ValueError("--seeds did not contain any integers.")
0193 |     return values
0194 | 
0195 | 
0196 | def quote_identifier(value: str) -> str:
0197 |     return '"' + value.replace('"', '""') + '"'
0198 | 
0199 | 
0200 | def normalise_space(value: Any) -> str:
0201 |     if value is None:
0202 |         return ""
0203 |     return re.sub(r"\s+", " ", str(value)).strip()
0204 | 
0205 | 
0206 | def normalise_sentence(value: str) -> str:
0207 |     text = normalise_space(value).lower()
0208 |     text = re.sub(r"[“”]", '"', text)
0209 |     text = re.sub(r"[‘’]", "'", text)
0210 |     return text
0211 | 
0212 | 
0213 | def tokenise(value: str) -> list[str]:
0214 |     return re.findall(r"\b[\w'-]+\b", value.lower())
0215 | 
0216 | 
0217 | def lexical_jaccard(left: str, right: str) -> float:
0218 |     a = set(tokenise(left))
0219 |     b = set(tokenise(right))
0220 |     if not a and not b:
0221 |         return 1.0
0222 |     if not a or not b:
0223 |         return 0.0
0224 |     return len(a & b) / len(a | b)
0225 | 
0226 | 
0227 | def stable_int_hash(*parts: Any) -> int:
0228 |     payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
0229 |     return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
0230 | 
0231 | 
0232 | def sentence_split(text: str) -> list[str]:
0233 |     """Dependency-free sentence splitter suitable for a first corpus probe."""
0234 |     cleaned = normalise_space(text)
0235 |     if not cleaned:
0236 |         return []
0237 | 
0238 |     # Split after likely sentence punctuation, or at paragraph/newline boundaries.
0239 |     pieces = re.split(
0240 |         r"(?<=[.!?])\s+(?=(?:[\"'“‘(\[]?[A-Z0-9]))|(?:\s*\n+\s*)",
0241 |         cleaned,
0242 |     )
0243 |     sentences = [normalise_space(piece) for piece in pieces]
0244 |     return [sentence for sentence in sentences if sentence]
0245 | 
0246 | 
0247 | def valid_sentence(
0248 |     sentence: str,
0249 |     *,
0250 |     min_chars: int,
0251 |     max_chars: int,
0252 | ) -> bool:
0253 |     length = len(sentence)
0254 |     if length < min_chars or length > max_chars:
0255 |         return False
0256 |     return len(tokenise(sentence)) >= 3
0257 | 
0258 | 
0259 | def log_loss_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
0260 |     p = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
0261 |     y = y_true.astype(float)
0262 |     return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
0263 | 
0264 | 
0265 | def brier_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
0266 |     return np.square(probabilities.astype(float) - y_true.astype(float))
0267 | 
0268 | 
0269 | # ---------------------------------------------------------------------------
0270 | # SQLite schema discovery
0271 | # ---------------------------------------------------------------------------
0272 | 
0273 | 
0274 | COLUMN_ALIASES = {
0275 |     "source": ["source", "publisher", "outlet"],
0276 |     "article_id": ["a_id", "article_id", "articleid", "story_id"],
0277 |     "version_id": ["version_id", "v_id", "version", "revision_id"],
0278 |     "text": ["text", "article_text", "body", "content"],
0279 |     "created": ["created", "created_at", "timestamp", "published_at", "date"],
0280 |     "title": ["title", "headline"],
0281 |     "num_versions": ["num_versions", "version_count", "n_versions"],
0282 | }
0283 | 
0284 | 
0285 | def sqlite_tables(connection: sqlite3.Connection) -> list[str]:
0286 |     rows = connection.execute(
0287 |         "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
0288 |     ).fetchall()
0289 |     return [str(row[0]) for row in rows]
0290 | 
0291 | 
0292 | def table_columns(
0293 |     connection: sqlite3.Connection,
0294 |     table: str,
0295 | ) -> list[str]:
0296 |     rows = connection.execute(
0297 |         f"PRAGMA table_info({quote_identifier(table)})"
0298 |     ).fetchall()
0299 |     return [str(row[1]) for row in rows]
0300 | 
0301 | 
0302 | def resolve_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
0303 |     lookup = {column.lower(): column for column in columns}
0304 |     for alias in aliases:
0305 |         if alias.lower() in lookup:
0306 |             return lookup[alias.lower()]
0307 |     return None
0308 | 
0309 | 
0310 | def discover_article_schema(
0311 |     connection: sqlite3.Connection,
0312 |     preferred_table: str | None,
0313 | ) -> ArticleSchema:
0314 |     tables = sqlite_tables(connection)
0315 |     if not tables:
0316 |         raise ValueError("The SQLite database contains no tables.")
0317 | 
0318 |     candidates = [preferred_table] if preferred_table else tables
0319 |     candidates = [table for table in candidates if table is not None]
0320 | 
0321 |     diagnostics: list[str] = []
0322 | 
0323 |     for table in candidates:
0324 |         if table not in tables:
0325 |             diagnostics.append(f"{table}: table not found")
0326 |             continue
0327 | 
0328 |         columns = table_columns(connection, table)
0329 |         resolved = {
0330 |             logical: resolve_column(columns, aliases)
0331 |             for logical, aliases in COLUMN_ALIASES.items()
0332 |         }
0333 | 
0334 |         required = ["source", "article_id", "version_id", "text"]
0335 |         missing = [name for name in required if resolved[name] is None]
0336 |         diagnostics.append(
0337 |             f"{table}: columns={columns}; missing_required={missing}"
0338 |         )
0339 |         if missing:
0340 |             continue
0341 | 
0342 |         return ArticleSchema(
0343 |             table=table,
0344 |             source=str(resolved["source"]),
0345 |             article_id=str(resolved["article_id"]),
0346 |             version_id=str(resolved["version_id"]),
0347 |             text=str(resolved["text"]),
0348 |             created=resolved["created"],
0349 |             title=resolved["title"],
0350 |             num_versions=resolved["num_versions"],
0351 |         )
0352 | 
0353 |     detail = "\n".join(f"  - {line}" for line in diagnostics)
0354 |     raise ValueError(
0355 |         "Could not find an article-version table with source, article ID, "
0356 |         f"version ID and text columns.\n{detail}"
0357 |     )
0358 | 
0359 | 
0360 | def inspect_database(
0361 |     connection: sqlite3.Connection,
0362 |     preferred_table: str | None,
0363 | ) -> None:
0364 |     print_header("SQLite database inspection")
0365 |     for table in sqlite_tables(connection):
0366 |         columns = table_columns(connection, table)
0367 |         try:
0368 |             count = connection.execute(
0369 |                 f"SELECT COUNT(*) FROM {quote_identifier(table)}"
0370 |             ).fetchone()[0]
0371 |         except sqlite3.DatabaseError:
0372 |             count = "unavailable"
0373 |         print(f"{table}: rows={count:,}" if isinstance(count, int) else f"{table}: rows={count}")
0374 |         print("  " + ", ".join(columns))
0375 | 
0376 |     print_header("Detected article schema")
0377 |     schema = discover_article_schema(connection, preferred_table)
0378 |     print(dataclasses.asdict(schema))
0379 | 
0380 | 
0381 | # ---------------------------------------------------------------------------
0382 | # Article sampling and version loading
0383 | # ---------------------------------------------------------------------------
0384 | 
0385 | 
0386 | def reservoir_sample_article_keys(
0387 |     connection: sqlite3.Connection,
0388 |     schema: ArticleSchema,
0389 |     *,
0390 |     max_articles: int,
0391 |     seed: int,
0392 |     sources: Sequence[str],
0393 | ) -> list[tuple[Any, Any, int]]:
0394 |     table = quote_identifier(schema.table)
0395 |     source_col = quote_identifier(schema.source)
0396 |     article_col = quote_identifier(schema.article_id)
0397 |     text_col = quote_identifier(schema.text)
0398 | 
0399 |     conditions = [
0400 |         f"{text_col} IS NOT NULL",
0401 |         f"LENGTH(TRIM({text_col})) > 0",
0402 |     ]
0403 |     params: list[Any] = []
0404 | 
0405 |     if sources:
0406 |         placeholders = ",".join("?" for _ in sources)
0407 |         conditions.append(f"CAST({source_col} AS TEXT) IN ({placeholders})")
0408 |         params.extend(sources)
0409 | 
0410 |     sql = f"""
0411 |         SELECT
0412 |             {source_col} AS source_value,
0413 |             {article_col} AS article_value,
0414 |             COUNT(*) AS version_count
0415 |         FROM {table}
0416 |         WHERE {' AND '.join(conditions)}
0417 |         GROUP BY {source_col}, {article_col}
0418 |         HAVING COUNT(*) >= 3
0419 |     """
0420 | 
0421 |     rng = random.Random(seed)
0422 |     reservoir: list[tuple[Any, Any, int]] = []
0423 |     seen = 0
0424 | 
0425 |     cursor = connection.execute(sql, params)
0426 |     for source_value, article_value, version_count in cursor:
0427 |         item = (source_value, article_value, int(version_count))
0428 |         seen += 1
0429 | 
0430 |         if max_articles <= 0:
0431 |             reservoir.append(item)
0432 |             continue
0433 | 
0434 |         if len(reservoir) < max_articles:
0435 |             reservoir.append(item)
0436 |         else:
0437 |             replacement = rng.randrange(seen)
0438 |             if replacement < max_articles:
0439 |                 reservoir[replacement] = item
0440 | 
0441 |     print(
0442 |         f"Qualifying articles with 3+ versions: {seen:,}; "
0443 |         f"selected: {len(reservoir):,}"
0444 |     )
0445 |     return reservoir
0446 | 
0447 | 
0448 | def load_selected_versions(
0449 |     connection: sqlite3.Connection,
0450 |     schema: ArticleSchema,
0451 |     selected_keys: Sequence[tuple[Any, Any, int]],
0452 | ) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
0453 |     if not selected_keys:
0454 |         return
0455 | 
0456 |     connection.execute("DROP TABLE IF EXISTS temp.pf_selected_articles")
0457 |     connection.execute(
0458 |         """
0459 |         CREATE TEMP TABLE pf_selected_articles (
0460 |             source_value,
0461 |             article_value,
0462 |             version_count INTEGER
0463 |         )
0464 |         """
0465 |     )
0466 |     connection.executemany(
0467 |         """
0468 |         INSERT INTO pf_selected_articles
0469 |             (source_value, article_value, version_count)
0470 |         VALUES (?, ?, ?)
0471 |         """,
0472 |         selected_keys,
0473 |     )
0474 | 
0475 |     table = quote_identifier(schema.table)
0476 |     source_col = quote_identifier(schema.source)
0477 |     article_col = quote_identifier(schema.article_id)
0478 |     version_col = quote_identifier(schema.version_id)
0479 |     text_col = quote_identifier(schema.text)
0480 | 
0481 |     created_expr = (
0482 |         f"a.{quote_identifier(schema.created)}"
0483 |         if schema.created
0484 |         else "NULL"
0485 |     )
0486 |     title_expr = (
0487 |         f"a.{quote_identifier(schema.title)}"
0488 |         if schema.title
0489 |         else "NULL"
0490 |     )
0491 | 
0492 |     sql = f"""
0493 |         SELECT
0494 |             a.{source_col} AS source_value,
0495 |             a.{article_col} AS article_value,
0496 |             a.{version_col} AS version_value,
0497 |             a.{text_col} AS text_value,
0498 |             {created_expr} AS created_value,
0499 |             {title_expr} AS title_value,
0500 |             k.version_count AS sampled_version_count
0501 |         FROM {table} AS a
0502 |         INNER JOIN temp.pf_selected_articles AS k
0503 |           ON a.{source_col} = k.source_value
0504 |          AND a.{article_col} = k.article_value
0505 |         WHERE a.{text_col} IS NOT NULL
0506 |           AND LENGTH(TRIM(a.{text_col})) > 0
0507 |         ORDER BY
0508 |             a.{source_col},
0509 |             a.{article_col},
0510 |             CASE WHEN created_value IS NULL THEN 1 ELSE 0 END,
0511 |             created_value,
0512 |             a.{version_col}
0513 |     """
0514 | 
0515 |     current_key: tuple[str, str] | None = None
0516 |     current_versions: list[dict[str, Any]] = []
0517 | 
0518 |     for row in connection.execute(sql):
0519 |         key = (str(row[0]), str(row[1]))
0520 |         version = {
0521 |             "source": str(row[0]),
0522 |             "article_id": str(row[1]),
0523 |             "version_id": str(row[2]),
0524 |             "text": str(row[3]),
0525 |             "created": None if row[4] is None else str(row[4]),
0526 |             "title": "" if row[5] is None else str(row[5]),
0527 |             "n_versions": int(row[6]),
0528 |         }
0529 | 
0530 |         if current_key is None:
0531 |             current_key = key
0532 | 
0533 |         if key != current_key:
0534 |             yield current_key, current_versions
0535 |             current_key = key
0536 |             current_versions = []
0537 | 
0538 |         current_versions.append(version)
0539 | 
0540 |     if current_key is not None and current_versions:
0541 |         yield current_key, current_versions
0542 | 
0543 | 
0544 | # ---------------------------------------------------------------------------
0545 | # Revision lineage extraction
0546 | # ---------------------------------------------------------------------------
0547 | 
0548 | 
0549 | def sentence_fate_map(
0550 |     middle_sentences: Sequence[str],
0551 |     future_sentences: Sequence[str],
0552 | ) -> dict[int, int]:
0553 |     """Return 0=unchanged in next version, 1=revised/removed."""
0554 |     middle_norm = [normalise_sentence(value) for value in middle_sentences]
0555 |     future_norm = [normalise_sentence(value) for value in future_sentences]
0556 | 
0557 |     matcher = difflib.SequenceMatcher(
0558 |         a=middle_norm,
0559 |         b=future_norm,
0560 |         autojunk=False,
0561 |     )
0562 | 
0563 |     fate: dict[int, int] = {}
0564 |     for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
0565 |         if tag == "equal":
0566 |             for index in range(i1, i2):
0567 |                 fate[index] = 0
0568 |         elif tag in {"replace", "delete"}:
0569 |             for index in range(i1, i2):
0570 |                 fate[index] = 1
0571 |         # Insertions add future material but do not change an existing V1 sentence.
0572 |     return fate
0573 | 
0574 | 
0575 | def extract_episodes_from_article(
0576 |     key: tuple[str, str],
0577 |     versions: Sequence[dict[str, Any]],
0578 |     *,
0579 |     context_before: int,
0580 |     context_after: int,
0581 |     min_sentence_chars: int,
0582 |     max_sentence_chars: int,
0583 |     min_edit_similarity: float,
0584 |     max_edit_similarity: float,
0585 | ) -> list[dict[str, Any]]:
0586 |     if len(versions) < 3:
0587 |         return []
0588 | 
0589 |     split_versions = [sentence_split(version["text"]) for version in versions]
0590 |     article_key = f"{key[0]}::{key[1]}"
0591 |     episodes: list[dict[str, Any]] = []
0592 | 
0593 |     for version_index in range(len(versions) - 2):
0594 |         old_version = versions[version_index]
0595 |         middle_version = versions[version_index + 1]
0596 |         future_version = versions[version_index + 2]
0597 | 
0598 |         old_sentences = split_versions[version_index]
0599 |         middle_sentences = split_versions[version_index + 1]
0600 |         future_sentences = split_versions[version_index + 2]
0601 | 
0602 |         if not old_sentences or not middle_sentences or not future_sentences:
0603 |             continue
0604 | 
0605 |         old_norm = [normalise_sentence(value) for value in old_sentences]
0606 |         middle_norm = [normalise_sentence(value) for value in middle_sentences]
0607 | 
0608 |         current_matcher = difflib.SequenceMatcher(
0609 |             a=old_norm,
0610 |             b=middle_norm,
0611 |             autojunk=False,
0612 |         )
0613 |         future_fate = sentence_fate_map(middle_sentences, future_sentences)
0614 | 
0615 |         for tag, i1, i2, j1, j2 in current_matcher.get_opcodes():
0616 |             # Start with clean one-to-one replacements. This gives a defensible
0617 |             # rejected/retained pair without ambiguous split/merge attribution.
0618 |             if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) != 1:
0619 |                 continue
0620 | 
0621 |             rejected = old_sentences[i1]
0622 |             retained = middle_sentences[j1]
0623 | 
0624 |             if not valid_sentence(
0625 |                 rejected,
0626 |                 min_chars=min_sentence_chars,
0627 |                 max_chars=max_sentence_chars,
0628 |             ):
0629 |                 continue
0630 |             if not valid_sentence(
0631 |                 retained,
0632 |                 min_chars=min_sentence_chars,
0633 |                 max_chars=max_sentence_chars,
0634 |             ):
0635 |                 continue
0636 |             if j1 not in future_fate:
0637 |                 continue
0638 | 
0639 |             similarity = difflib.SequenceMatcher(
0640 |                 a=normalise_sentence(rejected),
0641 |                 b=normalise_sentence(retained),
0642 |                 autojunk=False,
0643 |             ).ratio()
0644 |             if similarity < min_edit_similarity:
0645 |                 continue
0646 |             if similarity > max_edit_similarity:
0647 |                 continue
0648 | 
0649 |             before_start = max(0, j1 - context_before)
0650 |             after_end = min(len(middle_sentences), j1 + 1 + context_after)
0651 |             preceding = " ".join(middle_sentences[before_start:j1])
0652 |             following = " ".join(middle_sentences[j1 + 1:after_end])
0653 | 
0654 |             rejected_tokens = len(tokenise(rejected))
0655 |             retained_tokens = len(tokenise(retained))
0656 | 
0657 |             episodes.append(
0658 |                 {
0659 |                     "episode_id": (
0660 |                         f"{article_key}::{old_version['version_id']}"
0661 |                         f"->{middle_version['version_id']}::{j1}"
0662 |                     ),
0663 |                     "article_key": article_key,
0664 |                     "source": middle_version["source"],
0665 |                     "article_id": middle_version["article_id"],
0666 |                     "title": middle_version["title"],
0667 |                     "old_version_id": old_version["version_id"],
0668 |                     "retained_version_id": middle_version["version_id"],
0669 |                     "future_version_id": future_version["version_id"],
0670 |                     "version_index": version_index + 1,
0671 |                     "n_versions": len(versions),
0672 |                     "sentence_position": (
0673 |                         j1 / max(1, len(middle_sentences) - 1)
0674 |                     ),
0675 |                     "context_before": preceding,
0676 |                     "retained_sentence": retained,
0677 |                     "context_after": following,
0678 |                     "rejected_sentence": rejected,
0679 |                     "retained_chars": len(retained),
0680 |                     "rejected_chars": len(rejected),
0681 |                     "retained_tokens": retained_tokens,
0682 |                     "rejected_tokens": rejected_tokens,
0683 |                     "char_delta": len(retained) - len(rejected),
0684 |                     "token_delta": retained_tokens - rejected_tokens,
0685 |                     "edit_similarity": similarity,
0686 |                     "lexical_jaccard": lexical_jaccard(rejected, retained),
0687 |                     "revised_again_next_version": int(future_fate[j1]),
0688 |                 }
0689 |             )
0690 | 
0691 |     return episodes
0692 | 
0693 | 
0694 | def build_episode_dataframe(
0695 |     connection: sqlite3.Connection,
0696 |     schema: ArticleSchema,
0697 |     *,
0698 |     max_articles: int,
0699 |     max_episodes: int,
0700 |     sampling_seed: int,
0701 |     sources: Sequence[str],
0702 |     context_before: int,
0703 |     context_after: int,
0704 |     min_sentence_chars: int,
0705 |     max_sentence_chars: int,
0706 |     min_edit_similarity: float,
0707 |     max_edit_similarity: float,
0708 | ) -> pd.DataFrame:
0709 |     selected_keys = reservoir_sample_article_keys(
0710 |         connection,
0711 |         schema,
0712 |         max_articles=max_articles,
0713 |         seed=sampling_seed,
0714 |         sources=sources,
0715 |     )
0716 | 
0717 |     records: list[dict[str, Any]] = []
0718 |     processed_articles = 0
0719 | 
0720 |     for key, versions in load_selected_versions(connection, schema, selected_keys):
0721 |         processed_articles += 1
0722 |         article_records = extract_episodes_from_article(
0723 |             key,
0724 |             versions,
0725 |             context_before=context_before,
0726 |             context_after=context_after,
0727 |             min_sentence_chars=min_sentence_chars,
0728 |             max_sentence_chars=max_sentence_chars,
0729 |             min_edit_similarity=min_edit_similarity,
0730 |             max_edit_similarity=max_edit_similarity,
0731 |         )
0732 |         records.extend(article_records)
0733 | 
0734 |         if processed_articles % 1000 == 0:
0735 |             print(
0736 |                 f"Processed articles={processed_articles:,}; "
0737 |                 f"episodes={len(records):,}"
0738 |             )
0739 | 
0740 |         if max_episodes > 0 and len(records) >= max_episodes:
0741 |             records = records[:max_episodes]
0742 |             break
0743 | 
0744 |     if not records:
0745 |         raise ValueError(
0746 |             "No revision-lineage episodes were extracted. Run --inspect-only, "
0747 |             "check the article schema, or relax the edit/sentence filters."
0748 |         )
0749 | 
0750 |     frame = pd.DataFrame.from_records(records)
0751 |     frame = frame.drop_duplicates(subset=["episode_id"]).reset_index(drop=True)
0752 |     return frame
0753 | 
0754 | 
0755 | def save_episode_cache(frame: pd.DataFrame, path: Path) -> None:
0756 |     path.parent.mkdir(parents=True, exist_ok=True)
0757 |     suffixes = "".join(path.suffixes).lower()
0758 | 
0759 |     if suffixes.endswith(".parquet"):
0760 |         try:
0761 |             frame.to_parquet(path, index=False)
0762 |         except ImportError as exc:
0763 |             raise SystemExit(
0764 |                 "Saving Parquet requires pyarrow. Install it or use .csv.gz."
0765 |             ) from exc
0766 |     else:
0767 |         compression = "gzip" if suffixes.endswith(".gz") else None
0768 |         frame.to_csv(path, index=False, compression=compression)
0769 | 
0770 | 
0771 | def load_episode_cache(path: Path) -> pd.DataFrame:
0772 |     suffixes = "".join(path.suffixes).lower()
0773 |     if suffixes.endswith(".parquet"):
0774 |         return pd.read_parquet(path)
0775 |     return pd.read_csv(path)
0776 | 
0777 | 
0778 | # ---------------------------------------------------------------------------
0779 | # Feature sets and model
0780 | # ---------------------------------------------------------------------------
0781 | 
0782 | 
0783 | BASE_TEXT_COLUMNS = [
0784 |     "context_before",
0785 |     "retained_sentence",
0786 |     "context_after",
0787 | ]
0788 | PREFERENCE_TEXT_COLUMNS = ["rejected_sentence"]
0789 | 
0790 | BASE_NUMERIC_COLUMNS = [
0791 |     "version_index",
0792 |     "n_versions",
0793 |     "sentence_position",
0794 |     "retained_chars",
0795 |     "retained_tokens",
0796 | ]
0797 | PREFERENCE_NUMERIC_COLUMNS = [
0798 |     "rejected_chars",
0799 |     "rejected_tokens",
0800 |     "char_delta",
0801 |     "token_delta",
0802 |     "edit_similarity",
0803 |     "lexical_jaccard",
0804 | ]
0805 | BASE_CATEGORICAL_COLUMNS = ["source"]
0806 | 
0807 | PREFERENCE_BUNDLE_COLUMNS = (
0808 |     PREFERENCE_TEXT_COLUMNS + PREFERENCE_NUMERIC_COLUMNS
0809 | )
0810 | 
0811 | 
0812 | def build_model_pipeline(
0813 |     *,
0814 |     include_base: bool,
0815 |     include_preference: bool,
0816 |     seed: int,
0817 |     tfidf_max_features: int,
0818 | ):
0819 |     from sklearn.compose import ColumnTransformer
0820 |     from sklearn.impute import SimpleImputer
0821 |     from sklearn.linear_model import SGDClassifier
0822 |     from sklearn.pipeline import Pipeline
0823 |     from sklearn.preprocessing import OneHotEncoder, StandardScaler
0824 |     from sklearn.feature_extraction.text import TfidfVectorizer
0825 | 
0826 |     transformers: list[tuple[str, Any, Any]] = []
0827 | 
0828 |     if include_base:
0829 |         for column in BASE_TEXT_COLUMNS:
0830 |             transformers.append(
0831 |                 (
0832 |                     f"text_{column}",
0833 |                     TfidfVectorizer(
0834 |                         lowercase=True,
0835 |                         ngram_range=(1, 2),
0836 |                         min_df=1,
0837 |                         max_df=1.0,
0838 |                         max_features=tfidf_max_features,
0839 |                         sublinear_tf=True,
0840 |                     ),
0841 |                     column,
0842 |                 )
0843 |             )
0844 | 
0845 |         numeric_pipe = Pipeline(
0846 |             [
0847 |                 ("imputer", SimpleImputer(strategy="median")),
0848 |                 ("scale", StandardScaler(with_mean=False)),
0849 |             ]
0850 |         )
0851 |         transformers.append(("base_num", numeric_pipe, BASE_NUMERIC_COLUMNS))
0852 |         transformers.append(
0853 |             (
0854 |                 "base_cat",
0855 |                 OneHotEncoder(handle_unknown="ignore"),
0856 |                 BASE_CATEGORICAL_COLUMNS,
0857 |             )
0858 |         )
0859 | 
0860 |     if include_preference:
0861 |         transformers.append(
0862 |             (
0863 |                 "text_rejected",
0864 |                 TfidfVectorizer(
0865 |                     lowercase=True,
0866 |                     ngram_range=(1, 2),
0867 |                     min_df=1,
0868 |                     max_df=1.0,
0869 |                     max_features=tfidf_max_features,
0870 |                     sublinear_tf=True,
0871 |                 ),
0872 |                 "rejected_sentence",
0873 |             )
0874 |         )
0875 |         preference_numeric_pipe = Pipeline(
0876 |             [
0877 |                 ("imputer", SimpleImputer(strategy="median")),
0878 |                 ("scale", StandardScaler(with_mean=False)),
0879 |             ]
0880 |         )
0881 |         transformers.append(
0882 |             (
0883 |                 "preference_num",
0884 |                 preference_numeric_pipe,
0885 |                 PREFERENCE_NUMERIC_COLUMNS,
0886 |             )
0887 |         )
0888 | 
0889 |     preprocessor = ColumnTransformer(
0890 |         transformers=transformers,
0891 |         remainder="drop",
0892 |         sparse_threshold=0.1,
0893 |     )
0894 | 
0895 |     # Scalable probabilistic linear baseline. No class reweighting: PFI uses
0896 |     # proper probability scoring rules and must preserve observed prevalence.
0897 |     classifier = SGDClassifier(
0898 |         loss="log_loss",
0899 |         penalty="l2",
0900 |         alpha=1e-5,
0901 |         max_iter=2000,
0902 |         tol=1e-4,
0903 |         class_weight=None,
0904 |         random_state=seed,
0905 |         average=True,
0906 |     )
0907 | 
0908 |     return Pipeline(
0909 |         [
0910 |             ("preprocessor", preprocessor),
0911 |             ("classifier", classifier),
0912 |         ]
0913 |     )
0914 | 
0915 | 
0916 | def fit_and_score(
0917 |     train_df: pd.DataFrame,
0918 |     test_df: pd.DataFrame,
0919 |     *,
0920 |     include_base: bool,
0921 |     include_preference: bool,
0922 |     target_column: str,
0923 |     seed: int,
0924 |     tfidf_max_features: int,
0925 | ) -> tuple[np.ndarray, dict[str, float]]:
0926 |     from sklearn.metrics import (
0927 |         accuracy_score,
0928 |         average_precision_score,
0929 |         brier_score_loss,
0930 |         log_loss,
0931 |         roc_auc_score,
0932 |     )
0933 | 
0934 |     train_df = train_df.copy()
0935 |     test_df = test_df.copy()
0936 |     for text_column in BASE_TEXT_COLUMNS + PREFERENCE_TEXT_COLUMNS:
0937 |         if text_column in train_df.columns:
0938 |             train_df[text_column] = (
0939 |                 train_df[text_column]
0940 |                 .fillna("")
0941 |                 .astype(str)
0942 |                 .map(lambda value: value if value.strip() else "__empty__")
0943 |             )
0944 |             test_df[text_column] = (
0945 |                 test_df[text_column]
0946 |                 .fillna("")
0947 |                 .astype(str)
0948 |                 .map(lambda value: value if value.strip() else "__empty__")
0949 |             )
0950 | 
0951 |     y_train = train_df[target_column].astype(int).to_numpy()
0952 |     y_test = test_df[target_column].astype(int).to_numpy()
0953 | 
0954 |     if np.unique(y_train).size < 2:
0955 |         raise ValueError("The training split contains only one target class.")
0956 | 
0957 |     pipeline = build_model_pipeline(
0958 |         include_base=include_base,
0959 |         include_preference=include_preference,
0960 |         seed=seed,
0961 |         tfidf_max_features=tfidf_max_features,
0962 |     )
0963 |     pipeline.fit(train_df, y_train)
0964 | 
0965 |     probabilities = pipeline.predict_proba(test_df)[:, 1]
0966 |     predictions = (probabilities >= 0.5).astype(int)
0967 | 
0968 |     train_prevalence = float(y_train.mean())
0969 |     test_prevalence = float(y_test.mean())
0970 |     null_probabilities = np.full(
0971 |         len(y_test),
0972 |         np.clip(train_prevalence, 1e-12, 1.0 - 1e-12),
0973 |     )
0974 | 
0975 |     metrics = {
0976 |         "loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
0977 |         "brier": float(brier_score_loss(y_test, probabilities)),
0978 |         "auc": float(roc_auc_score(y_test, probabilities)),
0979 |         "average_precision": float(
0980 |             average_precision_score(y_test, probabilities)
0981 |         ),
0982 |         "accuracy": float(accuracy_score(y_test, predictions)),
0983 |         "train_prevalence": train_prevalence,
0984 |         "test_prevalence": test_prevalence,
0985 |         "mean_predicted_probability": float(probabilities.mean()),
0986 |         "null_log_loss": float(
0987 |             log_loss(y_test, null_probabilities, labels=[0, 1])
0988 |         ),
0989 |         "null_brier": float(
0990 |             brier_score_loss(y_test, null_probabilities)
0991 |         ),
0992 |     }
0993 |     return probabilities, metrics
0994 | 
0995 | 
0996 | def shuffle_preference_bundle(
0997 |     frame: pd.DataFrame,
0998 |     *,
0999 |     seed: int,
1000 | ) -> pd.DataFrame:
1001 |     """Shuffle rejected-sentence evidence within source/length buckets."""
1002 |     shuffled = frame.copy()
1003 |     shuffled["_length_bucket"] = (
1004 |         shuffled["retained_chars"].fillna(0).astype(int) // 80
1005 |     ).clip(upper=20)
1006 | 
1007 |     pieces: list[pd.DataFrame] = []
1008 |     grouped = shuffled.groupby(
1009 |         ["source", "_length_bucket"],
1010 |         dropna=False,
1011 |         sort=False,
1012 |     )
1013 | 
1014 |     for group_index, (_key, part) in enumerate(grouped):
1015 |         if len(part) <= 1:
1016 |             values = part[PREFERENCE_BUNDLE_COLUMNS].to_numpy()
1017 |         else:
1018 |             values = part[PREFERENCE_BUNDLE_COLUMNS].sample(
1019 |                 frac=1.0,
1020 |                 random_state=seed + group_index,
1021 |             ).to_numpy()
1022 |         pieces.append(
1023 |             pd.DataFrame(
1024 |                 values,
1025 |                 columns=PREFERENCE_BUNDLE_COLUMNS,
1026 |                 index=part.index,
1027 |             )
1028 |         )
1029 | 
1030 |     replacement = pd.concat(pieces).sort_index()
1031 |     for column in PREFERENCE_BUNDLE_COLUMNS:
1032 |         shuffled[column] = replacement[column].to_numpy()
1033 |     for column in PREFERENCE_NUMERIC_COLUMNS:
1034 |         shuffled[column] = pd.to_numeric(
1035 |             shuffled[column],
1036 |             errors="coerce",
1037 |         )
1038 |     return shuffled.drop(columns=["_length_bucket"])
1039 | 
1040 | 
1041 | def build_group_loss_stats(
1042 |     test_df: pd.DataFrame,
1043 |     *,
1044 |     target_column: str,
1045 |     seed: int,
1046 |     predictions: dict[str, np.ndarray],
1047 | ) -> pd.DataFrame:
1048 |     y_true = test_df[target_column].astype(int).to_numpy()
1049 |     row_stats = pd.DataFrame(
1050 |         {
1051 |             "group_id": test_df["article_key"].astype(str).to_numpy(),
1052 |             "n_rows": 1,
1053 |         },
1054 |         index=test_df.index,
1055 |     )
1056 | 
1057 |     for name, probability in predictions.items():
1058 |         row_stats[f"{name}_log_sum"] = log_loss_components(
1059 |             y_true, probability
1060 |         )
1061 |         row_stats[f"{name}_brier_sum"] = brier_components(
1062 |             y_true, probability
1063 |         )
1064 | 
1065 |     grouped = row_stats.groupby("group_id", as_index=False).sum(
1066 |         numeric_only=True
1067 |     )
1068 |     grouped["seed"] = seed
1069 |     return grouped
1070 | 
1071 | 
1072 | def evaluate_seed(
1073 |     frame: pd.DataFrame,
1074 |     *,
1075 |     seed: int,
1076 |     test_fraction: float,
1077 |     target_column: str,
1078 |     tfidf_max_features: int,
1079 | ) -> EvaluationBundle:
1080 |     rng = np.random.default_rng(seed)
1081 |     groups = np.asarray(
1082 |         sorted(frame["article_key"].dropna().astype(str).unique())
1083 |     )
1084 |     if len(groups) < 2:
1085 |         raise ValueError("At least two article groups are required.")
1086 | 
1087 |     rng.shuffle(groups)
1088 |     n_test = max(1, int(round(len(groups) * test_fraction)))
1089 |     n_test = min(n_test, len(groups) - 1)
1090 |     test_groups = set(groups[:n_test])
1091 | 
1092 |     group_values = frame["article_key"].astype(str)
1093 |     train_df = frame[~group_values.isin(test_groups)].copy()
1094 |     test_df = frame[group_values.isin(test_groups)].copy()
1095 | 
1096 |     configurations = {
1097 |         "context_retained_no_preference": (True, False),
1098 |         "rejected_preference_only": (False, True),
1099 |         "context_retained_plus_preference": (True, True),
1100 |     }
1101 | 
1102 |     rows: list[ResultRow] = []
1103 |     predictions: dict[str, np.ndarray] = {}
1104 | 
1105 |     for feature_set, (include_base, include_preference) in configurations.items():
1106 |         probability, metrics = fit_and_score(
1107 |             train_df,
1108 |             test_df,
1109 |             include_base=include_base,
1110 |             include_preference=include_preference,
1111 |             target_column=target_column,
1112 |             seed=seed,
1113 |             tfidf_max_features=tfidf_max_features,
1114 |         )
1115 |         predictions[feature_set] = probability
1116 |         rows.append(
1117 |             ResultRow(
1118 |                 track="newsedits",
1119 |                 condition="sentence_revision_stability",
1120 |                 seed=seed,
1121 |                 target=target_column,
1122 |                 feature_set=feature_set,
1123 |                 n_train=len(train_df),
1124 |                 n_test=len(test_df),
1125 |                 n_train_groups=train_df["article_key"].nunique(),
1126 |                 n_test_groups=test_df["article_key"].nunique(),
1127 |                 **metrics,
1128 |             )
1129 |         )
1130 | 
1131 |     shuffled_train = shuffle_preference_bundle(
1132 |         train_df,
1133 |         seed=seed + 10_000,
1134 |     )
1135 |     shuffled_test = shuffle_preference_bundle(
1136 |         test_df,
1137 |         seed=seed + 20_000,
1138 |     )
1139 |     shuffled_name = "context_retained_plus_shuffled_preference"
1140 |     shuffled_probability, shuffled_metrics = fit_and_score(
1141 |         shuffled_train,
1142 |         shuffled_test,
1143 |         include_base=True,
1144 |         include_preference=True,
1145 |         target_column=target_column,
1146 |         seed=seed,
1147 |         tfidf_max_features=tfidf_max_features,
1148 |     )
1149 |     predictions[shuffled_name] = shuffled_probability
1150 |     rows.append(
1151 |         ResultRow(
1152 |             track="newsedits",
1153 |             condition="sentence_revision_stability",
1154 |             seed=seed,
1155 |             target=target_column,
1156 |             feature_set=shuffled_name,
1157 |             n_train=len(shuffled_train),
1158 |             n_test=len(shuffled_test),
1159 |             n_train_groups=shuffled_train["article_key"].nunique(),
1160 |             n_test_groups=shuffled_test["article_key"].nunique(),
1161 |             **shuffled_metrics,
1162 |         )
1163 |     )
1164 | 
1165 |     stats = build_group_loss_stats(
1166 |         test_df,
1167 |         target_column=target_column,
1168 |         seed=seed,
1169 |         predictions={
1170 |             "no_pref": predictions["context_retained_no_preference"],
1171 |             "full": predictions["context_retained_plus_preference"],
1172 |             "shuffled": predictions[shuffled_name],
1173 |         },
1174 |     )
1175 |     return EvaluationBundle(rows=rows, group_loss_stats=stats)
1176 | 
1177 | 
1178 | # ---------------------------------------------------------------------------
1179 | # Aggregation and bootstrap
1180 | # ---------------------------------------------------------------------------
1181 | 
1182 | 
1183 | def hierarchical_bootstrap_interval(
1184 |     stats: pd.DataFrame,
1185 |     *,
1186 |     statistic: str,
1187 |     samples: int,
1188 |     confidence_level: float,
1189 |     seed: int,
1190 | ) -> tuple[float, float]:
1191 |     if samples <= 0:
1192 |         return float("nan"), float("nan")
1193 | 
1194 |     rng = np.random.default_rng(seed)
1195 |     seed_values = np.asarray(sorted(stats["seed"].unique()), dtype=int)
1196 | 
1197 |     columns = [
1198 |         "n_rows",
1199 |         "no_pref_log_sum",
1200 |         "full_log_sum",
1201 |         "shuffled_log_sum",
1202 |         "no_pref_brier_sum",
1203 |         "full_brier_sum",
1204 |         "shuffled_brier_sum",
1205 |     ]
1206 |     by_seed = {
1207 |         int(seed_value): stats.loc[
1208 |             stats["seed"] == seed_value, columns
1209 |         ].to_numpy(dtype=float)
1210 |         for seed_value in seed_values
1211 |     }
1212 | 
1213 |     def compute(totals: np.ndarray) -> float:
1214 |         n_rows = totals[0]
1215 |         if statistic == "pfi_log_loss":
1216 |             return float((totals[1] - totals[2]) / n_rows)
1217 |         if statistic == "shuffle_gap_log_loss":
1218 |             return float((totals[3] - totals[2]) / n_rows)
1219 |         if statistic == "pfi_brier":
1220 |             return float((totals[4] - totals[5]) / n_rows)
1221 |         if statistic == "shuffle_gap_brier":
1222 |             return float((totals[6] - totals[5]) / n_rows)
1223 |         raise ValueError(statistic)
1224 | 
1225 |     draws = np.empty(samples, dtype=float)
1226 |     for draw_index in range(samples):
1227 |         sampled_seeds = rng.choice(
1228 |             seed_values,
1229 |             size=len(seed_values),
1230 |             replace=True,
1231 |         )
1232 |         totals = np.zeros(len(columns), dtype=float)
1233 |         for sampled_seed in sampled_seeds:
1234 |             array = by_seed[int(sampled_seed)]
1235 |             indices = rng.integers(0, len(array), size=len(array))
1236 |             totals += array[indices].sum(axis=0)
1237 |         draws[draw_index] = compute(totals)
1238 | 
1239 |     alpha = 1.0 - confidence_level
1240 |     return (
1241 |         float(np.quantile(draws, alpha / 2.0)),
1242 |         float(np.quantile(draws, 1.0 - alpha / 2.0)),
1243 |     )
1244 | 
1245 | 
1246 | def build_summary_rows(
1247 |     rows: list[ResultRow],
1248 |     group_stats: pd.DataFrame,
1249 |     *,
1250 |     bootstrap_samples: int,
1251 |     confidence_level: float,
1252 |     bootstrap_seed: int,
1253 | ) -> list[SummaryRow]:
1254 |     frame = pd.DataFrame([dataclasses.asdict(row) for row in rows])
1255 |     seed_values = sorted(frame["seed"].unique())
1256 | 
1257 |     values_by_statistic: dict[str, list[float]] = {
1258 |         "pfi_log_loss": [],
1259 |         "pfi_brier": [],
1260 |         "shuffle_gap_log_loss": [],
1261 |         "shuffle_gap_brier": [],
1262 |     }
1263 | 
1264 |     for seed in seed_values:
1265 |         indexed = frame[frame["seed"] == seed].set_index("feature_set")
1266 |         no_pref = indexed.loc["context_retained_no_preference"]
1267 |         full = indexed.loc["context_retained_plus_preference"]
1268 |         shuffled = indexed.loc[
1269 |             "context_retained_plus_shuffled_preference"
1270 |         ]
1271 | 
1272 |         values_by_statistic["pfi_log_loss"].append(
1273 |             float(no_pref["loss"] - full["loss"])
1274 |         )
1275 |         values_by_statistic["pfi_brier"].append(
1276 |             float(no_pref["brier"] - full["brier"])
1277 |         )
1278 |         values_by_statistic["shuffle_gap_log_loss"].append(
1279 |             float(shuffled["loss"] - full["loss"])
1280 |         )
1281 |         values_by_statistic["shuffle_gap_brier"].append(
1282 |             float(shuffled["brier"] - full["brier"])
1283 |         )
1284 | 
1285 |     summaries: list[SummaryRow] = []
1286 |     for statistic, values_list in values_by_statistic.items():
1287 |         values = np.asarray(values_list, dtype=float)
1288 |         low, high = hierarchical_bootstrap_interval(
1289 |             group_stats,
1290 |             statistic=statistic,
1291 |             samples=bootstrap_samples,
1292 |             confidence_level=confidence_level,
1293 |             seed=bootstrap_seed + sum(ord(char) for char in statistic),
1294 |         )
1295 |         summaries.append(
1296 |             SummaryRow(
1297 |                 track="newsedits",
1298 |                 condition="sentence_revision_stability",
1299 |                 statistic=statistic,
1300 |                 n_seeds=len(values),
1301 |                 mean=float(values.mean()),
1302 |                 seed_std=(
1303 |                     float(values.std(ddof=1)) if len(values) > 1 else 0.0
1304 |                 ),
1305 |                 ci_low=low,
1306 |                 ci_high=high,
1307 |                 positive_seeds=int((values > 0).sum()),
1308 |                 confidence_level=confidence_level,
1309 |                 bootstrap_samples=bootstrap_samples,
1310 |             )
1311 |         )
1312 |     return summaries
1313 | 
1314 | 
1315 | # ---------------------------------------------------------------------------
1316 | # Reporting
1317 | # ---------------------------------------------------------------------------
1318 | 
1319 | 
1320 | def audit_episodes(frame: pd.DataFrame) -> None:
1321 |     print_header("NewsEdits revision-lineage audit")
1322 |     print(f"Episodes: {len(frame):,}")
1323 |     print(f"Articles: {frame['article_key'].nunique():,}")
1324 |     print(f"Sources: {frame['source'].nunique():,}")
1325 |     print(
1326 |         "Target revised-again rate: "
1327 |         f"{frame['revised_again_next_version'].mean():.6f}"
1328 |     )
1329 |     print("\nTarget counts:")
1330 |     print(
1331 |         frame["revised_again_next_version"]
1332 |         .value_counts(dropna=False)
1333 |         .sort_index()
1334 |         .to_string()
1335 |     )
1336 |     print("\nLargest sources:")
1337 |     print(frame["source"].value_counts().head(20).to_string())
1338 |     print("\nEdit similarity:")
1339 |     print(frame["edit_similarity"].describe().to_string())
1340 | 
1341 | 
1342 | def print_results(
1343 |     rows: list[ResultRow],
1344 |     summaries: list[SummaryRow],
1345 | ) -> None:
1346 |     result_frame = pd.DataFrame(
1347 |         [dataclasses.asdict(row) for row in rows]
1348 |     )
1349 |     aggregate = result_frame.groupby("feature_set")[
1350 |         [
1351 |             "loss",
1352 |             "brier",
1353 |             "auc",
1354 |             "average_precision",
1355 |             "accuracy",
1356 |             "mean_predicted_probability",
1357 |         ]
1358 |     ].agg(["mean", "std"])
1359 | 
1360 |     print_header("Feature-set metrics across seeds")
1361 |     print(aggregate.to_string(float_format=lambda value: f"{value:.6f}"))
1362 | 
1363 |     summary_frame = pd.DataFrame(
1364 |         [dataclasses.asdict(row) for row in summaries]
1365 |     )
1366 |     print_header("PFI and shuffled-control summary")
1367 |     print(
1368 |         summary_frame[
1369 |             [
1370 |                 "statistic",
1371 |                 "n_seeds",
1372 |                 "mean",
1373 |                 "seed_std",
1374 |                 "ci_low",
1375 |                 "ci_high",
1376 |                 "positive_seeds",
1377 |             ]
1378 |         ].to_string(
1379 |             index=False,
1380 |             float_format=lambda value: f"{value:.6f}",
1381 |         )
1382 |     )
1383 | 
1384 | 
1385 | # ---------------------------------------------------------------------------
1386 | # CLI
1387 | # ---------------------------------------------------------------------------
1388 | 
1389 | 
1390 | def parse_args(argv: Sequence[str]) -> argparse.Namespace:
1391 |     parser = argparse.ArgumentParser(
1392 |         description=(
1393 |             "Test whether rejected sentence alternatives improve forecasts "
1394 |             "of later sentence revision in NewsEdits."
1395 |         )
1396 |     )
1397 |     parser.add_argument("--db", help="Path to the NewsEdits SQLite database.")
1398 |     parser.add_argument(
1399 |         "--articles-table",
1400 |         default=None,
1401 |         help="Optional article-version table name; otherwise auto-discovered.",
1402 |     )
1403 |     parser.add_argument(
1404 |         "--inspect-only",
1405 |         action="store_true",
1406 |         help="Print SQLite tables/schema and exit.",
1407 |     )
1408 |     parser.add_argument(
1409 |         "--episode-cache",
1410 |         default=None,
1411 |         help=(
1412 |             "CSV, CSV.GZ or Parquet path. Existing cache is loaded unless "
1413 |             "--rebuild-cache is supplied."
1414 |         ),
1415 |     )
1416 |     parser.add_argument(
1417 |         "--rebuild-cache",
1418 |         action="store_true",
1419 |     )
1420 |     parser.add_argument(
1421 |         "--max-articles",
1422 |         type=int,
1423 |         default=10_000,
1424 |         help="Reservoir-sampled articles with 3+ versions. Use 0 for all.",
1425 |     )
1426 |     parser.add_argument(
1427 |         "--max-episodes",
1428 |         type=int,
1429 |         default=0,
1430 |         help="Stop after this many episodes; 0 means no episode cap.",
1431 |     )
1432 |     parser.add_argument(
1433 |         "--sampling-seed",
1434 |         type=int,
1435 |         default=1729,
1436 |     )
1437 |     parser.add_argument(
1438 |         "--source",
1439 |         action="append",
1440 |         default=[],
1441 |         help="Optional source/outlet filter; may be repeated.",
1442 |     )
1443 |     parser.add_argument(
1444 |         "--context-before",
1445 |         type=int,
1446 |         default=2,
1447 |     )
1448 |     parser.add_argument(
1449 |         "--context-after",
1450 |         type=int,
1451 |         default=1,
1452 |     )
1453 |     parser.add_argument(
1454 |         "--min-sentence-chars",
1455 |         type=int,
1456 |         default=25,
1457 |     )
1458 |     parser.add_argument(
1459 |         "--max-sentence-chars",
1460 |         type=int,
1461 |         default=600,
1462 |     )
1463 |     parser.add_argument(
1464 |         "--min-edit-similarity",
1465 |         type=float,
1466 |         default=0.20,
1467 |     )
1468 |     parser.add_argument(
1469 |         "--max-edit-similarity",
1470 |         type=float,
1471 |         default=0.98,
1472 |     )
1473 |     parser.add_argument("--seed", type=int, default=7)
1474 |     parser.add_argument(
1475 |         "--seeds",
1476 |         default=None,
1477 |         help="Comma-separated split/model seeds.",
1478 |     )
1479 |     parser.add_argument(
1480 |         "--test-fraction",
1481 |         type=float,
1482 |         default=0.2,
1483 |     )
1484 |     parser.add_argument(
1485 |         "--tfidf-max-features",
1486 |         type=int,
1487 |         default=40_000,
1488 |         help="Maximum TF-IDF features per text field.",
1489 |     )
1490 |     parser.add_argument(
1491 |         "--bootstrap-samples",
1492 |         type=int,
1493 |         default=2000,
1494 |     )
1495 |     parser.add_argument(
1496 |         "--bootstrap-seed",
1497 |         type=int,
1498 |         default=2718,
1499 |     )
1500 |     parser.add_argument(
1501 |         "--confidence-level",
1502 |         type=float,
1503 |         default=0.95,
1504 |     )
1505 |     parser.add_argument(
1506 |         "--out",
1507 |         default=None,
1508 |         help="Per-seed result CSV.",
1509 |     )
1510 |     parser.add_argument(
1511 |         "--summary-out",
1512 |         default=None,
1513 |         help="PFI summary CSV.",
1514 |     )
1515 |     return parser.parse_args(argv)
1516 | 
1517 | 
1518 | def main(argv: Sequence[str]) -> int:
1519 |     args = parse_args(argv)
1520 | 
1521 |     if not 0.0 < args.test_fraction < 1.0:
1522 |         raise SystemExit("--test-fraction must be between 0 and 1.")
1523 |     if not 0.0 <= args.min_edit_similarity < args.max_edit_similarity <= 1.0:
1524 |         raise SystemExit("Edit similarity bounds must satisfy 0 <= min < max <= 1.")
1525 |     if args.bootstrap_samples < 0:
1526 |         raise SystemExit("--bootstrap-samples must be non-negative.")
1527 | 
1528 |     cache_path = Path(args.episode_cache) if args.episode_cache else None
1529 | 
1530 |     if cache_path and cache_path.exists() and not args.rebuild_cache:
1531 |         print(f"Loading episode cache: {cache_path}")
1532 |         episodes = load_episode_cache(cache_path)
1533 |     else:
1534 |         if not args.db:
1535 |             raise SystemExit(
1536 |                 "--db is required when an episode cache is not available."
1537 |             )
1538 | 
1539 |         db_path = Path(args.db)
1540 |         if not db_path.exists():
1541 |             raise SystemExit(f"SQLite database not found: {db_path}")
1542 | 
1543 |         connection = sqlite3.connect(str(db_path))
1544 |         try:
1545 |             if args.inspect_only:
1546 |                 inspect_database(connection, args.articles_table)
1547 |                 return 0
1548 | 
1549 |             schema = discover_article_schema(
1550 |                 connection,
1551 |                 args.articles_table,
1552 |             )
1553 |             print_header("Detected article schema")
1554 |             print(dataclasses.asdict(schema))
1555 | 
1556 |             episodes = build_episode_dataframe(
1557 |                 connection,
1558 |                 schema,
1559 |                 max_articles=args.max_articles,
1560 |                 max_episodes=args.max_episodes,
1561 |                 sampling_seed=args.sampling_seed,
1562 |                 sources=args.source,
1563 |                 context_before=args.context_before,
1564 |                 context_after=args.context_after,
1565 |                 min_sentence_chars=args.min_sentence_chars,
1566 |                 max_sentence_chars=args.max_sentence_chars,
1567 |                 min_edit_similarity=args.min_edit_similarity,
1568 |                 max_edit_similarity=args.max_edit_similarity,
1569 |             )
1570 |         finally:
1571 |             connection.close()
1572 | 
1573 |         if cache_path:
1574 |             save_episode_cache(episodes, cache_path)
1575 |             print(f"Saved episode cache: {cache_path}")
1576 | 
1577 |     required_columns = {
1578 |         "article_key",
1579 |         "source",
1580 |         "context_before",
1581 |         "retained_sentence",
1582 |         "context_after",
1583 |         "rejected_sentence",
1584 |         "revised_again_next_version",
1585 |         *BASE_NUMERIC_COLUMNS,
1586 |         *PREFERENCE_NUMERIC_COLUMNS,
1587 |     }
1588 |     missing = sorted(required_columns - set(episodes.columns))
1589 |     if missing:
1590 |         raise SystemExit(
1591 |             f"Episode cache is missing required columns: {missing}"
1592 |         )
1593 | 
1594 |     audit_episodes(episodes)
1595 | 
1596 |     seeds = parse_int_list(args.seeds, args.seed)
1597 |     all_rows: list[ResultRow] = []
1598 |     all_stats: list[pd.DataFrame] = []
1599 | 
1600 |     for seed in seeds:
1601 |         print(f"\nEvaluating seed {seed}...")
1602 |         bundle = evaluate_seed(
1603 |             episodes,
1604 |             seed=seed,
1605 |             test_fraction=args.test_fraction,
1606 |             target_column="revised_again_next_version",
1607 |             tfidf_max_features=args.tfidf_max_features,
1608 |         )
1609 |         all_rows.extend(bundle.rows)
1610 |         all_stats.append(bundle.group_loss_stats)
1611 | 
1612 |     combined_stats = pd.concat(all_stats, ignore_index=True)
1613 |     summaries = build_summary_rows(
1614 |         all_rows,
1615 |         combined_stats,
1616 |         bootstrap_samples=args.bootstrap_samples,
1617 |         confidence_level=args.confidence_level,
1618 |         bootstrap_seed=args.bootstrap_seed,
1619 |     )
1620 | 
1621 |     print_results(all_rows, summaries)
1622 | 
1623 |     if args.out:
1624 |         output = pd.DataFrame(
1625 |             [dataclasses.asdict(row) for row in all_rows]
1626 |         )
1627 |         output.to_csv(args.out, index=False)
1628 |         print(f"\nSaved per-seed results to {args.out}")
1629 | 
1630 |     if args.summary_out:
1631 |         summary_output = pd.DataFrame(
1632 |             [dataclasses.asdict(row) for row in summaries]
1633 |         )
1634 |         summary_output.to_csv(args.summary_out, index=False)
1635 |         print(f"Saved summary results to {args.summary_out}")
1636 | 
1637 |     return 0
1638 | 
1639 | 
1640 | if __name__ == "__main__":
1641 |     raise SystemExit(main(sys.argv[1:]))
```


---

## F0002 — `newsedits_ablation.py`

```text
FILE_ID: F0002
PATH: newsedits_ablation.py
LANGUAGE: python
LINES: 2337
BYTES_UTF8: 73821
SHA256: e64e0ef7c3f2ab7c66c43174ff0021568383bf24ce6708ee5513a890773fce7c
```

```python
0001 | #!/usr/bin/env python3
0002 | r"""
0003 | PreferenceFutures — NewsEdits mechanism-ablation probe.
0004 | 
0005 | Research question
0006 | -----------------
0007 | 
0008 | Does a real human revision event contain incremental information about the
0009 | future of the revised sentence?
0010 | 
0011 | For three consecutive article versions V0 -> V1 -> V2:
0012 | 
0013 |     rejected sentence: sentence in V0
0014 |     retained sentence: its one-to-one replacement in V1
0015 |     future: whether the retained V1 sentence is revised or removed in V2
0016 | 
0017 | The baseline already sees the retained sentence and its V1 context:
0018 | 
0019 |     P(F | context, retained sentence, metadata)
0020 | 
0021 | The preference-informed model additionally sees what the journalist replaced:
0022 | 
0023 |     P(F | context, retained sentence, rejected sentence, edit evidence, metadata)
0024 | 
0025 | Preference Future Information:
0026 | 
0027 |     PFI = Loss(baseline) - Loss(preference-informed)
0028 | 
0029 | Positive held-out PFI means the linked preference bundle carries information
0030 | about the next revision beyond the retained sentence itself. This v4 probe
0031 | separates rejected-text semantics, edit geometry and lexical relationship.
0032 | 
0033 | This is a revealed-revision-preference experiment, not an explicit A/B-vote
0034 | experiment and not a causal estimate.
0035 | 
0036 | Expected NewsEdits source
0037 | -------------------------
0038 | 
0039 | The official NewsEdits download provides source-specific compressed SQLite
0040 | databases such as ``nyt-matched-sentences.db.gz``. After decompression, the
0041 | database normally contains ``split_sentences`` and ``matched_sentences``.
0042 | This script reads ``split_sentences`` directly and reconstructs complete
0043 | article versions from ``entry_id``, ``version``, ``sent_idx`` and ``sentence``.
0044 | 
0045 | A full-article table with SOURCE, A_ID, VERSION_ID and TEXT is also supported
0046 | for compatible exports.
0047 | 
0048 | Dependencies
0049 | ------------
0050 | 
0051 |     pip install pandas numpy scikit-learn
0052 | 
0053 | Optional, only for Parquet episode caches:
0054 | 
0055 |     pip install pyarrow
0056 | 
0057 | Examples
0058 | --------
0059 | 
0060 | Inspect the database schema:
0061 | 
0062 |     python preference_futures_newsedits_v4_ablation.py \
0063 |       --db /path/to/newsedits.db \
0064 |       --inspect-only
0065 | 
0066 | Windows PowerShell smoke test:
0067 | 
0068 |     python preference_futures_newsedits_v4_ablation.py `
0069 |       --db C:\data\newsedits.db `
0070 |       --max-articles 5000 `
0071 |       --max-episodes 50000 `
0072 |       --seeds 1,2,3 `
0073 |       --bootstrap-samples 500 `
0074 |       --episode-cache newsedits_smoke_episodes.csv.gz `
0075 |       --ablation-profile core `
0076 |       --out newsedits_ablation_smoke_runs.csv `
0077 |       --summary-out newsedits_ablation_smoke_summary.csv
0078 | 
0079 | Larger run:
0080 | 
0081 |     python preference_futures_newsedits_v4_ablation.py `
0082 |       --db C:\data\newsedits.db `
0083 |       --max-articles 100000 `
0084 |       --seeds 1,2,3,4,5,6,7,8,9,10 `
0085 |       --bootstrap-samples 5000 `
0086 |       --episode-cache newsedits_full_episodes.csv.gz `
0087 |       --ablation-profile full `
0088 |       --out newsedits_ablation_full_runs.csv `
0089 |       --summary-out newsedits_ablation_full_summary.csv
0090 | 
0091 | Use --max-articles 0 to request every qualifying article. Build and validate a
0092 | smaller cache first: the complete corpus is very large.
0093 | """
0094 | 
0095 | from __future__ import annotations
0096 | 
0097 | import argparse
0098 | import dataclasses
0099 | import difflib
0100 | import hashlib
0101 | import math
0102 | import random
0103 | import re
0104 | import sqlite3
0105 | import sys
0106 | from pathlib import Path
0107 | from typing import Any, Iterable, Iterator, Sequence
0108 | 
0109 | import numpy as np
0110 | import pandas as pd
0111 | 
0112 | 
0113 | # ---------------------------------------------------------------------------
0114 | # Data records
0115 | # ---------------------------------------------------------------------------
0116 | 
0117 | 
0118 | @dataclasses.dataclass(frozen=True)
0119 | class ArticleSchema:
0120 |     table: str
0121 |     source: str
0122 |     article_id: str
0123 |     version_id: str
0124 |     text: str
0125 |     created: str | None
0126 |     title: str | None
0127 |     num_versions: str | None
0128 | 
0129 | 
0130 | @dataclasses.dataclass(frozen=True)
0131 | class SplitSentenceSchema:
0132 |     table: str
0133 |     article_id: str
0134 |     version_id: str
0135 |     sentence_id: str
0136 |     sentence: str
0137 | 
0138 | 
0139 | @dataclasses.dataclass
0140 | class ResultRow:
0141 |     track: str
0142 |     condition: str
0143 |     seed: int
0144 |     target: str
0145 |     feature_set: str
0146 |     n_train: int
0147 |     n_test: int
0148 |     n_train_groups: int
0149 |     n_test_groups: int
0150 |     loss: float
0151 |     brier: float
0152 |     auc: float
0153 |     average_precision: float
0154 |     accuracy: float
0155 |     train_prevalence: float
0156 |     test_prevalence: float
0157 |     mean_predicted_probability: float
0158 |     probability_min: float
0159 |     probability_p01: float
0160 |     probability_p05: float
0161 |     probability_median: float
0162 |     probability_p95: float
0163 |     probability_p99: float
0164 |     probability_max: float
0165 |     calibration_gap: float
0166 |     solver: str
0167 |     converged: bool
0168 |     n_iter: int
0169 |     null_log_loss: float
0170 |     null_brier: float
0171 | 
0172 | 
0173 | @dataclasses.dataclass
0174 | class SummaryRow:
0175 |     track: str
0176 |     condition: str
0177 |     comparison: str
0178 |     metric: str
0179 |     reference_feature_set: str
0180 |     candidate_feature_set: str
0181 |     n_seeds: int
0182 |     mean_gain: float
0183 |     seed_std: float
0184 |     ci_low: float
0185 |     ci_high: float
0186 |     positive_seeds: int
0187 |     confidence_level: float
0188 |     bootstrap_samples: int
0189 | 
0190 | 
0191 | @dataclasses.dataclass
0192 | class EvaluationBundle:
0193 |     rows: list[ResultRow]
0194 |     group_loss_stats: pd.DataFrame
0195 | 
0196 | 
0197 | # ---------------------------------------------------------------------------
0198 | # General utilities
0199 | # ---------------------------------------------------------------------------
0200 | 
0201 | 
0202 | def print_header(title: str) -> None:
0203 |     print("\n" + "=" * 100)
0204 |     print(title)
0205 |     print("=" * 100)
0206 | 
0207 | 
0208 | def parse_int_list(value: str | None, fallback: int) -> list[int]:
0209 |     if value is None or not value.strip():
0210 |         return [fallback]
0211 |     values = [int(part.strip()) for part in value.split(",") if part.strip()]
0212 |     if not values:
0213 |         raise ValueError("--seeds did not contain any integers.")
0214 |     return values
0215 | 
0216 | 
0217 | def quote_identifier(value: str) -> str:
0218 |     return '"' + value.replace('"', '""') + '"'
0219 | 
0220 | 
0221 | def normalise_space(value: Any) -> str:
0222 |     if value is None:
0223 |         return ""
0224 |     return re.sub(r"\s+", " ", str(value)).strip()
0225 | 
0226 | 
0227 | def normalise_sentence(value: str) -> str:
0228 |     text = normalise_space(value).lower()
0229 |     text = re.sub(r"[“”]", '"', text)
0230 |     text = re.sub(r"[‘’]", "'", text)
0231 |     return text
0232 | 
0233 | 
0234 | def tokenise(value: str) -> list[str]:
0235 |     return re.findall(r"\b[\w'-]+\b", value.lower())
0236 | 
0237 | 
0238 | def lexical_jaccard(left: str, right: str) -> float:
0239 |     a = set(tokenise(left))
0240 |     b = set(tokenise(right))
0241 |     if not a and not b:
0242 |         return 1.0
0243 |     if not a or not b:
0244 |         return 0.0
0245 |     return len(a & b) / len(a | b)
0246 | 
0247 | 
0248 | def stable_int_hash(*parts: Any) -> int:
0249 |     payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
0250 |     return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
0251 | 
0252 | 
0253 | def sentence_split(text: str) -> list[str]:
0254 |     """Dependency-free sentence splitter suitable for a first corpus probe."""
0255 |     cleaned = normalise_space(text)
0256 |     if not cleaned:
0257 |         return []
0258 | 
0259 |     # Split after likely sentence punctuation, or at paragraph/newline boundaries.
0260 |     pieces = re.split(
0261 |         r"(?<=[.!?])\s+(?=(?:[\"'“‘(\[]?[A-Z0-9]))|(?:\s*\n+\s*)",
0262 |         cleaned,
0263 |     )
0264 |     sentences = [normalise_space(piece) for piece in pieces]
0265 |     return [sentence for sentence in sentences if sentence]
0266 | 
0267 | 
0268 | def valid_sentence(
0269 |     sentence: str,
0270 |     *,
0271 |     min_chars: int,
0272 |     max_chars: int,
0273 | ) -> bool:
0274 |     length = len(sentence)
0275 |     if length < min_chars or length > max_chars:
0276 |         return False
0277 |     return len(tokenise(sentence)) >= 3
0278 | 
0279 | 
0280 | def log_loss_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
0281 |     p = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
0282 |     y = y_true.astype(float)
0283 |     return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
0284 | 
0285 | 
0286 | def brier_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
0287 |     return np.square(probabilities.astype(float) - y_true.astype(float))
0288 | 
0289 | 
0290 | # ---------------------------------------------------------------------------
0291 | # SQLite schema discovery
0292 | # ---------------------------------------------------------------------------
0293 | 
0294 | 
0295 | COLUMN_ALIASES = {
0296 |     "source": ["source", "publisher", "outlet"],
0297 |     "article_id": ["a_id", "article_id", "articleid", "story_id"],
0298 |     "version_id": ["version_id", "v_id", "version", "revision_id"],
0299 |     "text": ["text", "article_text", "body", "content"],
0300 |     "created": ["created", "created_at", "timestamp", "published_at", "date"],
0301 |     "title": ["title", "headline"],
0302 |     "num_versions": ["num_versions", "version_count", "n_versions"],
0303 | }
0304 | 
0305 | 
0306 | SPLIT_COLUMN_ALIASES = {
0307 |     "article_id": ["entry_id", "a_id", "article_id", "articleid", "story_id"],
0308 |     "version_id": ["version", "version_id", "v_id", "revision_id"],
0309 |     "sentence_id": ["sent_idx", "sentence_id", "sent_id", "sentence_index"],
0310 |     "sentence": ["sentence", "sent", "text"],
0311 | }
0312 | 
0313 | 
0314 | def sqlite_tables(connection: sqlite3.Connection) -> list[str]:
0315 |     rows = connection.execute(
0316 |         "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
0317 |     ).fetchall()
0318 |     return [str(row[0]) for row in rows]
0319 | 
0320 | 
0321 | def table_columns(
0322 |     connection: sqlite3.Connection,
0323 |     table: str,
0324 | ) -> list[str]:
0325 |     rows = connection.execute(
0326 |         f"PRAGMA table_info({quote_identifier(table)})"
0327 |     ).fetchall()
0328 |     return [str(row[1]) for row in rows]
0329 | 
0330 | 
0331 | def resolve_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
0332 |     lookup = {column.lower(): column for column in columns}
0333 |     for alias in aliases:
0334 |         if alias.lower() in lookup:
0335 |             return lookup[alias.lower()]
0336 |     return None
0337 | 
0338 | 
0339 | def discover_article_schema(
0340 |     connection: sqlite3.Connection,
0341 |     preferred_table: str | None,
0342 | ) -> ArticleSchema:
0343 |     tables = sqlite_tables(connection)
0344 |     if not tables:
0345 |         raise ValueError("The SQLite database contains no tables.")
0346 | 
0347 |     candidates = [preferred_table] if preferred_table else tables
0348 |     candidates = [table for table in candidates if table is not None]
0349 | 
0350 |     diagnostics: list[str] = []
0351 | 
0352 |     for table in candidates:
0353 |         if table not in tables:
0354 |             diagnostics.append(f"{table}: table not found")
0355 |             continue
0356 | 
0357 |         columns = table_columns(connection, table)
0358 |         resolved = {
0359 |             logical: resolve_column(columns, aliases)
0360 |             for logical, aliases in COLUMN_ALIASES.items()
0361 |         }
0362 | 
0363 |         required = ["source", "article_id", "version_id", "text"]
0364 |         missing = [name for name in required if resolved[name] is None]
0365 |         diagnostics.append(
0366 |             f"{table}: columns={columns}; missing_required={missing}"
0367 |         )
0368 |         if missing:
0369 |             continue
0370 | 
0371 |         return ArticleSchema(
0372 |             table=table,
0373 |             source=str(resolved["source"]),
0374 |             article_id=str(resolved["article_id"]),
0375 |             version_id=str(resolved["version_id"]),
0376 |             text=str(resolved["text"]),
0377 |             created=resolved["created"],
0378 |             title=resolved["title"],
0379 |             num_versions=resolved["num_versions"],
0380 |         )
0381 | 
0382 |     detail = "\n".join(f"  - {line}" for line in diagnostics)
0383 |     raise ValueError(
0384 |         "Could not find an article-version table with source, article ID, "
0385 |         f"version ID and text columns.\n{detail}"
0386 |     )
0387 | 
0388 | 
0389 | def discover_split_sentence_schema(
0390 |     connection: sqlite3.Connection,
0391 |     preferred_table: str | None = None,
0392 | ) -> SplitSentenceSchema:
0393 |     """Discover the official NewsEdits ``split_sentences`` table schema."""
0394 |     tables = sqlite_tables(connection)
0395 |     if not tables:
0396 |         raise ValueError("The SQLite database contains no tables.")
0397 | 
0398 |     if preferred_table:
0399 |         candidates = [preferred_table]
0400 |     else:
0401 |         candidates = [
0402 |             table
0403 |             for table in tables
0404 |             if table.lower() == "split_sentences"
0405 |         ] + [
0406 |             table
0407 |             for table in tables
0408 |             if table.lower() != "split_sentences"
0409 |         ]
0410 | 
0411 |     diagnostics: list[str] = []
0412 |     for table in candidates:
0413 |         if table not in tables:
0414 |             diagnostics.append(f"{table}: table not found")
0415 |             continue
0416 | 
0417 |         columns = table_columns(connection, table)
0418 |         resolved = {
0419 |             logical: resolve_column(columns, aliases)
0420 |             for logical, aliases in SPLIT_COLUMN_ALIASES.items()
0421 |         }
0422 |         missing = [
0423 |             logical for logical, value in resolved.items()
0424 |             if value is None
0425 |         ]
0426 |         diagnostics.append(
0427 |             f"{table}: columns={columns}; missing_required={missing}"
0428 |         )
0429 |         if missing:
0430 |             continue
0431 | 
0432 |         return SplitSentenceSchema(
0433 |             table=table,
0434 |             article_id=str(resolved["article_id"]),
0435 |             version_id=str(resolved["version_id"]),
0436 |             sentence_id=str(resolved["sentence_id"]),
0437 |             sentence=str(resolved["sentence"]),
0438 |         )
0439 | 
0440 |     detail = "\n".join(f"  - {line}" for line in diagnostics)
0441 |     raise ValueError(
0442 |         "Could not find an official NewsEdits split-sentence table with "
0443 |         "entry/article ID, version, sentence index and sentence text.\n"
0444 |         f"{detail}"
0445 |     )
0446 | 
0447 | 
0448 | def inspect_database(
0449 |     connection: sqlite3.Connection,
0450 |     preferred_table: str | None,
0451 |     preferred_split_table: str | None = None,
0452 | ) -> None:
0453 |     print_header("SQLite database inspection")
0454 |     for table in sqlite_tables(connection):
0455 |         columns = table_columns(connection, table)
0456 |         try:
0457 |             count = connection.execute(
0458 |                 f"SELECT COUNT(*) FROM {quote_identifier(table)}"
0459 |             ).fetchone()[0]
0460 |         except sqlite3.DatabaseError:
0461 |             count = "unavailable"
0462 |         print(f"{table}: rows={count:,}" if isinstance(count, int) else f"{table}: rows={count}")
0463 |         print("  " + ", ".join(columns))
0464 | 
0465 |     article_error: str | None = None
0466 |     split_error: str | None = None
0467 | 
0468 |     try:
0469 |         schema = discover_article_schema(connection, preferred_table)
0470 |         print_header("Detected full-article schema")
0471 |         print(dataclasses.asdict(schema))
0472 |     except ValueError as exc:
0473 |         article_error = str(exc)
0474 | 
0475 |     try:
0476 |         split_schema = discover_split_sentence_schema(
0477 |             connection,
0478 |             preferred_split_table,
0479 |         )
0480 |         print_header("Detected official split-sentence schema")
0481 |         print(dataclasses.asdict(split_schema))
0482 |     except ValueError as exc:
0483 |         split_error = str(exc)
0484 | 
0485 |     if article_error and split_error:
0486 |         raise ValueError(
0487 |             "No supported NewsEdits schema was detected.\n\n"
0488 |             f"Full-article attempt:\n{article_error}\n\n"
0489 |             f"Split-sentence attempt:\n{split_error}"
0490 |         )
0491 | 
0492 | 
0493 | # ---------------------------------------------------------------------------
0494 | # Article sampling and version loading
0495 | # ---------------------------------------------------------------------------
0496 | 
0497 | 
0498 | def reservoir_sample_article_keys(
0499 |     connection: sqlite3.Connection,
0500 |     schema: ArticleSchema,
0501 |     *,
0502 |     max_articles: int,
0503 |     seed: int,
0504 |     sources: Sequence[str],
0505 | ) -> list[tuple[Any, Any, int]]:
0506 |     table = quote_identifier(schema.table)
0507 |     source_col = quote_identifier(schema.source)
0508 |     article_col = quote_identifier(schema.article_id)
0509 |     text_col = quote_identifier(schema.text)
0510 | 
0511 |     conditions = [
0512 |         f"{text_col} IS NOT NULL",
0513 |         f"LENGTH(TRIM({text_col})) > 0",
0514 |     ]
0515 |     params: list[Any] = []
0516 | 
0517 |     if sources:
0518 |         placeholders = ",".join("?" for _ in sources)
0519 |         conditions.append(f"CAST({source_col} AS TEXT) IN ({placeholders})")
0520 |         params.extend(sources)
0521 | 
0522 |     sql = f"""
0523 |         SELECT
0524 |             {source_col} AS source_value,
0525 |             {article_col} AS article_value,
0526 |             COUNT(*) AS version_count
0527 |         FROM {table}
0528 |         WHERE {' AND '.join(conditions)}
0529 |         GROUP BY {source_col}, {article_col}
0530 |         HAVING COUNT(*) >= 3
0531 |     """
0532 | 
0533 |     rng = random.Random(seed)
0534 |     reservoir: list[tuple[Any, Any, int]] = []
0535 |     seen = 0
0536 | 
0537 |     cursor = connection.execute(sql, params)
0538 |     for source_value, article_value, version_count in cursor:
0539 |         item = (source_value, article_value, int(version_count))
0540 |         seen += 1
0541 | 
0542 |         if max_articles <= 0:
0543 |             reservoir.append(item)
0544 |             continue
0545 | 
0546 |         if len(reservoir) < max_articles:
0547 |             reservoir.append(item)
0548 |         else:
0549 |             replacement = rng.randrange(seen)
0550 |             if replacement < max_articles:
0551 |                 reservoir[replacement] = item
0552 | 
0553 |     print(
0554 |         f"Qualifying articles with 3+ versions: {seen:,}; "
0555 |         f"selected: {len(reservoir):,}"
0556 |     )
0557 |     return reservoir
0558 | 
0559 | 
0560 | def load_selected_versions(
0561 |     connection: sqlite3.Connection,
0562 |     schema: ArticleSchema,
0563 |     selected_keys: Sequence[tuple[Any, Any, int]],
0564 | ) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
0565 |     if not selected_keys:
0566 |         return
0567 | 
0568 |     connection.execute("DROP TABLE IF EXISTS temp.pf_selected_articles")
0569 |     connection.execute(
0570 |         """
0571 |         CREATE TEMP TABLE pf_selected_articles (
0572 |             source_value,
0573 |             article_value,
0574 |             version_count INTEGER
0575 |         )
0576 |         """
0577 |     )
0578 |     connection.executemany(
0579 |         """
0580 |         INSERT INTO pf_selected_articles
0581 |             (source_value, article_value, version_count)
0582 |         VALUES (?, ?, ?)
0583 |         """,
0584 |         selected_keys,
0585 |     )
0586 | 
0587 |     table = quote_identifier(schema.table)
0588 |     source_col = quote_identifier(schema.source)
0589 |     article_col = quote_identifier(schema.article_id)
0590 |     version_col = quote_identifier(schema.version_id)
0591 |     text_col = quote_identifier(schema.text)
0592 | 
0593 |     created_expr = (
0594 |         f"a.{quote_identifier(schema.created)}"
0595 |         if schema.created
0596 |         else "NULL"
0597 |     )
0598 |     title_expr = (
0599 |         f"a.{quote_identifier(schema.title)}"
0600 |         if schema.title
0601 |         else "NULL"
0602 |     )
0603 | 
0604 |     sql = f"""
0605 |         SELECT
0606 |             a.{source_col} AS source_value,
0607 |             a.{article_col} AS article_value,
0608 |             a.{version_col} AS version_value,
0609 |             a.{text_col} AS text_value,
0610 |             {created_expr} AS created_value,
0611 |             {title_expr} AS title_value,
0612 |             k.version_count AS sampled_version_count
0613 |         FROM {table} AS a
0614 |         INNER JOIN temp.pf_selected_articles AS k
0615 |           ON a.{source_col} = k.source_value
0616 |          AND a.{article_col} = k.article_value
0617 |         WHERE a.{text_col} IS NOT NULL
0618 |           AND LENGTH(TRIM(a.{text_col})) > 0
0619 |         ORDER BY
0620 |             a.{source_col},
0621 |             a.{article_col},
0622 |             CASE WHEN created_value IS NULL THEN 1 ELSE 0 END,
0623 |             created_value,
0624 |             a.{version_col}
0625 |     """
0626 | 
0627 |     current_key: tuple[str, str] | None = None
0628 |     current_versions: list[dict[str, Any]] = []
0629 | 
0630 |     for row in connection.execute(sql):
0631 |         key = (str(row[0]), str(row[1]))
0632 |         version = {
0633 |             "source": str(row[0]),
0634 |             "article_id": str(row[1]),
0635 |             "version_id": str(row[2]),
0636 |             "text": str(row[3]),
0637 |             "created": None if row[4] is None else str(row[4]),
0638 |             "title": "" if row[5] is None else str(row[5]),
0639 |             "n_versions": int(row[6]),
0640 |         }
0641 | 
0642 |         if current_key is None:
0643 |             current_key = key
0644 | 
0645 |         if key != current_key:
0646 |             yield current_key, current_versions
0647 |             current_key = key
0648 |             current_versions = []
0649 | 
0650 |         current_versions.append(version)
0651 | 
0652 |     if current_key is not None and current_versions:
0653 |         yield current_key, current_versions
0654 | 
0655 | 
0656 | 
0657 | # ---------------------------------------------------------------------------
0658 | # Official NewsEdits split-sentence loading
0659 | # ---------------------------------------------------------------------------
0660 | 
0661 | 
0662 | def infer_source_name(db_path: Path, explicit_source: str | None) -> str:
0663 |     if explicit_source:
0664 |         return explicit_source
0665 | 
0666 |     name = db_path.name
0667 |     for suffix in [".db.gz", ".sqlite.gz", ".sqlite3.gz", ".db", ".sqlite", ".sqlite3"]:
0668 |         if name.lower().endswith(suffix):
0669 |             name = name[: -len(suffix)]
0670 |             break
0671 |     name = re.sub(
0672 |         r"[-_](matched[-_]?sentences|sentence[-_]?diffs|processed)$",
0673 |         "",
0674 |         name,
0675 |         flags=re.IGNORECASE,
0676 |     )
0677 |     return name or "unknown"
0678 | 
0679 | 
0680 | def reservoir_sample_split_article_ids(
0681 |     connection: sqlite3.Connection,
0682 |     schema: SplitSentenceSchema,
0683 |     *,
0684 |     max_articles: int,
0685 |     seed: int,
0686 | ) -> list[tuple[Any, int]]:
0687 |     table = quote_identifier(schema.table)
0688 |     article_col = quote_identifier(schema.article_id)
0689 |     version_col = quote_identifier(schema.version_id)
0690 |     sentence_col = quote_identifier(schema.sentence)
0691 | 
0692 |     sql = f"""
0693 |         SELECT
0694 |             {article_col} AS article_value,
0695 |             COUNT(DISTINCT {version_col}) AS version_count
0696 |         FROM {table}
0697 |         WHERE {sentence_col} IS NOT NULL
0698 |           AND LENGTH(TRIM({sentence_col})) > 0
0699 |         GROUP BY {article_col}
0700 |         HAVING COUNT(DISTINCT {version_col}) >= 3
0701 |     """
0702 | 
0703 |     rng = random.Random(seed)
0704 |     reservoir: list[tuple[Any, int]] = []
0705 |     seen = 0
0706 | 
0707 |     for article_value, version_count in connection.execute(sql):
0708 |         item = (article_value, int(version_count))
0709 |         seen += 1
0710 | 
0711 |         if max_articles <= 0:
0712 |             reservoir.append(item)
0713 |         elif len(reservoir) < max_articles:
0714 |             reservoir.append(item)
0715 |         else:
0716 |             replacement = rng.randrange(seen)
0717 |             if replacement < max_articles:
0718 |                 reservoir[replacement] = item
0719 | 
0720 |     print(
0721 |         f"Qualifying entries with 3+ versions: {seen:,}; "
0722 |         f"selected: {len(reservoir):,}"
0723 |     )
0724 |     return reservoir
0725 | 
0726 | 
0727 | def load_selected_split_versions(
0728 |     connection: sqlite3.Connection,
0729 |     schema: SplitSentenceSchema,
0730 |     selected_ids: Sequence[tuple[Any, int]],
0731 |     *,
0732 |     source_name: str,
0733 | ) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
0734 |     if not selected_ids:
0735 |         return
0736 | 
0737 |     connection.execute("DROP TABLE IF EXISTS temp.pf_selected_entries")
0738 |     connection.execute(
0739 |         """
0740 |         CREATE TEMP TABLE pf_selected_entries (
0741 |             article_value,
0742 |             version_count INTEGER
0743 |         )
0744 |         """
0745 |     )
0746 |     connection.executemany(
0747 |         """
0748 |         INSERT INTO pf_selected_entries
0749 |             (article_value, version_count)
0750 |         VALUES (?, ?)
0751 |         """,
0752 |         selected_ids,
0753 |     )
0754 | 
0755 |     table = quote_identifier(schema.table)
0756 |     article_col = quote_identifier(schema.article_id)
0757 |     version_col = quote_identifier(schema.version_id)
0758 |     sentence_id_col = quote_identifier(schema.sentence_id)
0759 |     sentence_col = quote_identifier(schema.sentence)
0760 | 
0761 |     sql = f"""
0762 |         SELECT
0763 |             s.{article_col} AS article_value,
0764 |             s.{version_col} AS version_value,
0765 |             s.{sentence_id_col} AS sentence_index,
0766 |             s.{sentence_col} AS sentence_value,
0767 |             k.version_count AS sampled_version_count
0768 |         FROM {table} AS s
0769 |         INNER JOIN temp.pf_selected_entries AS k
0770 |           ON s.{article_col} = k.article_value
0771 |         WHERE s.{sentence_col} IS NOT NULL
0772 |           AND LENGTH(TRIM(s.{sentence_col})) > 0
0773 |         ORDER BY
0774 |             s.{article_col},
0775 |             CAST(s.{version_col} AS REAL),
0776 |             s.{version_col},
0777 |             CAST(s.{sentence_id_col} AS INTEGER),
0778 |             s.{sentence_id_col}
0779 |     """
0780 | 
0781 |     current_article: str | None = None
0782 |     current_version: str | None = None
0783 |     current_sentences: list[str] = []
0784 |     versions: list[dict[str, Any]] = []
0785 |     sampled_version_count = 0
0786 | 
0787 |     def flush_version() -> None:
0788 |         nonlocal current_version, current_sentences, versions
0789 |         if current_version is None or not current_sentences:
0790 |             return
0791 |         versions.append(
0792 |             {
0793 |                 "source": source_name,
0794 |                 "article_id": str(current_article),
0795 |                 "version_id": str(current_version),
0796 |                 "text": " ".join(current_sentences),
0797 |                 "created": None,
0798 |                 "title": "",
0799 |                 "n_versions": sampled_version_count,
0800 |             }
0801 |         )
0802 |         current_sentences = []
0803 | 
0804 |     for (
0805 |         article_value,
0806 |         version_value,
0807 |         _sentence_index,
0808 |         sentence_value,
0809 |         version_count,
0810 |     ) in connection.execute(sql):
0811 |         article_value_str = str(article_value)
0812 |         version_value_str = str(version_value)
0813 | 
0814 |         if current_article is None:
0815 |             current_article = article_value_str
0816 |             current_version = version_value_str
0817 |             sampled_version_count = int(version_count)
0818 | 
0819 |         if article_value_str != current_article:
0820 |             flush_version()
0821 |             yield (source_name, current_article), versions
0822 |             current_article = article_value_str
0823 |             current_version = version_value_str
0824 |             current_sentences = []
0825 |             versions = []
0826 |             sampled_version_count = int(version_count)
0827 |         elif version_value_str != current_version:
0828 |             flush_version()
0829 |             current_version = version_value_str
0830 | 
0831 |         sentence = normalise_space(sentence_value)
0832 |         if sentence:
0833 |             current_sentences.append(sentence)
0834 | 
0835 |     if current_article is not None:
0836 |         flush_version()
0837 |         if versions:
0838 |             yield (source_name, current_article), versions
0839 | 
0840 | 
0841 | def build_episode_dataframe_from_split_sentences(
0842 |     connection: sqlite3.Connection,
0843 |     schema: SplitSentenceSchema,
0844 |     *,
0845 |     source_name: str,
0846 |     max_articles: int,
0847 |     max_episodes: int,
0848 |     sampling_seed: int,
0849 |     context_before: int,
0850 |     context_after: int,
0851 |     min_sentence_chars: int,
0852 |     max_sentence_chars: int,
0853 |     min_edit_similarity: float,
0854 |     max_edit_similarity: float,
0855 | ) -> pd.DataFrame:
0856 |     selected_ids = reservoir_sample_split_article_ids(
0857 |         connection,
0858 |         schema,
0859 |         max_articles=max_articles,
0860 |         seed=sampling_seed,
0861 |     )
0862 | 
0863 |     records: list[dict[str, Any]] = []
0864 |     processed_articles = 0
0865 | 
0866 |     for key, versions in load_selected_split_versions(
0867 |         connection,
0868 |         schema,
0869 |         selected_ids,
0870 |         source_name=source_name,
0871 |     ):
0872 |         processed_articles += 1
0873 |         records.extend(
0874 |             extract_episodes_from_article(
0875 |                 key,
0876 |                 versions,
0877 |                 context_before=context_before,
0878 |                 context_after=context_after,
0879 |                 min_sentence_chars=min_sentence_chars,
0880 |                 max_sentence_chars=max_sentence_chars,
0881 |                 min_edit_similarity=min_edit_similarity,
0882 |                 max_edit_similarity=max_edit_similarity,
0883 |             )
0884 |         )
0885 | 
0886 |         if processed_articles % 1000 == 0:
0887 |             print(
0888 |                 f"Processed articles={processed_articles:,}; "
0889 |                 f"episodes={len(records):,}"
0890 |             )
0891 | 
0892 |         if max_episodes > 0 and len(records) >= max_episodes:
0893 |             records = records[:max_episodes]
0894 |             break
0895 | 
0896 |     if not records:
0897 |         raise ValueError(
0898 |             "No revision-lineage episodes were extracted from split_sentences. "
0899 |             "Try more articles or relax the edit-similarity filters."
0900 |         )
0901 | 
0902 |     frame = pd.DataFrame.from_records(records)
0903 |     return frame.drop_duplicates(subset=["episode_id"]).reset_index(drop=True)
0904 | 
0905 | 
0906 | # ---------------------------------------------------------------------------
0907 | # Revision lineage extraction
0908 | # ---------------------------------------------------------------------------
0909 | 
0910 | 
0911 | def sentence_fate_map(
0912 |     middle_sentences: Sequence[str],
0913 |     future_sentences: Sequence[str],
0914 | ) -> dict[int, int]:
0915 |     """Return 0=unchanged in next version, 1=revised/removed."""
0916 |     middle_norm = [normalise_sentence(value) for value in middle_sentences]
0917 |     future_norm = [normalise_sentence(value) for value in future_sentences]
0918 | 
0919 |     matcher = difflib.SequenceMatcher(
0920 |         a=middle_norm,
0921 |         b=future_norm,
0922 |         autojunk=False,
0923 |     )
0924 | 
0925 |     fate: dict[int, int] = {}
0926 |     for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
0927 |         if tag == "equal":
0928 |             for index in range(i1, i2):
0929 |                 fate[index] = 0
0930 |         elif tag in {"replace", "delete"}:
0931 |             for index in range(i1, i2):
0932 |                 fate[index] = 1
0933 |         # Insertions add future material but do not change an existing V1 sentence.
0934 |     return fate
0935 | 
0936 | 
0937 | def extract_episodes_from_article(
0938 |     key: tuple[str, str],
0939 |     versions: Sequence[dict[str, Any]],
0940 |     *,
0941 |     context_before: int,
0942 |     context_after: int,
0943 |     min_sentence_chars: int,
0944 |     max_sentence_chars: int,
0945 |     min_edit_similarity: float,
0946 |     max_edit_similarity: float,
0947 | ) -> list[dict[str, Any]]:
0948 |     if len(versions) < 3:
0949 |         return []
0950 | 
0951 |     split_versions = [sentence_split(version["text"]) for version in versions]
0952 |     article_key = f"{key[0]}::{key[1]}"
0953 |     episodes: list[dict[str, Any]] = []
0954 | 
0955 |     for version_index in range(len(versions) - 2):
0956 |         old_version = versions[version_index]
0957 |         middle_version = versions[version_index + 1]
0958 |         future_version = versions[version_index + 2]
0959 | 
0960 |         old_sentences = split_versions[version_index]
0961 |         middle_sentences = split_versions[version_index + 1]
0962 |         future_sentences = split_versions[version_index + 2]
0963 | 
0964 |         if not old_sentences or not middle_sentences or not future_sentences:
0965 |             continue
0966 | 
0967 |         old_norm = [normalise_sentence(value) for value in old_sentences]
0968 |         middle_norm = [normalise_sentence(value) for value in middle_sentences]
0969 | 
0970 |         current_matcher = difflib.SequenceMatcher(
0971 |             a=old_norm,
0972 |             b=middle_norm,
0973 |             autojunk=False,
0974 |         )
0975 |         future_fate = sentence_fate_map(middle_sentences, future_sentences)
0976 | 
0977 |         for tag, i1, i2, j1, j2 in current_matcher.get_opcodes():
0978 |             # Start with clean one-to-one replacements. This gives a defensible
0979 |             # rejected/retained pair without ambiguous split/merge attribution.
0980 |             if tag != "replace" or (i2 - i1) != 1 or (j2 - j1) != 1:
0981 |                 continue
0982 | 
0983 |             rejected = old_sentences[i1]
0984 |             retained = middle_sentences[j1]
0985 | 
0986 |             if not valid_sentence(
0987 |                 rejected,
0988 |                 min_chars=min_sentence_chars,
0989 |                 max_chars=max_sentence_chars,
0990 |             ):
0991 |                 continue
0992 |             if not valid_sentence(
0993 |                 retained,
0994 |                 min_chars=min_sentence_chars,
0995 |                 max_chars=max_sentence_chars,
0996 |             ):
0997 |                 continue
0998 |             if j1 not in future_fate:
0999 |                 continue
1000 | 
1001 |             similarity = difflib.SequenceMatcher(
1002 |                 a=normalise_sentence(rejected),
1003 |                 b=normalise_sentence(retained),
1004 |                 autojunk=False,
1005 |             ).ratio()
1006 |             if similarity < min_edit_similarity:
1007 |                 continue
1008 |             if similarity > max_edit_similarity:
1009 |                 continue
1010 | 
1011 |             before_start = max(0, j1 - context_before)
1012 |             after_end = min(len(middle_sentences), j1 + 1 + context_after)
1013 |             preceding = " ".join(middle_sentences[before_start:j1])
1014 |             following = " ".join(middle_sentences[j1 + 1:after_end])
1015 | 
1016 |             rejected_tokens = len(tokenise(rejected))
1017 |             retained_tokens = len(tokenise(retained))
1018 | 
1019 |             episodes.append(
1020 |                 {
1021 |                     "episode_id": (
1022 |                         f"{article_key}::{old_version['version_id']}"
1023 |                         f"->{middle_version['version_id']}::{j1}"
1024 |                     ),
1025 |                     "article_key": article_key,
1026 |                     "source": middle_version["source"],
1027 |                     "article_id": middle_version["article_id"],
1028 |                     "title": middle_version["title"],
1029 |                     "old_version_id": old_version["version_id"],
1030 |                     "retained_version_id": middle_version["version_id"],
1031 |                     "future_version_id": future_version["version_id"],
1032 |                     "version_index": version_index + 1,
1033 |                     "n_versions": len(versions),
1034 |                     "sentence_position": (
1035 |                         j1 / max(1, len(middle_sentences) - 1)
1036 |                     ),
1037 |                     "context_before": preceding,
1038 |                     "retained_sentence": retained,
1039 |                     "context_after": following,
1040 |                     "rejected_sentence": rejected,
1041 |                     "retained_chars": len(retained),
1042 |                     "rejected_chars": len(rejected),
1043 |                     "retained_tokens": retained_tokens,
1044 |                     "rejected_tokens": rejected_tokens,
1045 |                     "char_delta": len(retained) - len(rejected),
1046 |                     "token_delta": retained_tokens - rejected_tokens,
1047 |                     "edit_similarity": similarity,
1048 |                     "lexical_jaccard": lexical_jaccard(rejected, retained),
1049 |                     "revised_again_next_version": int(future_fate[j1]),
1050 |                 }
1051 |             )
1052 | 
1053 |     return episodes
1054 | 
1055 | 
1056 | def build_episode_dataframe(
1057 |     connection: sqlite3.Connection,
1058 |     schema: ArticleSchema,
1059 |     *,
1060 |     max_articles: int,
1061 |     max_episodes: int,
1062 |     sampling_seed: int,
1063 |     sources: Sequence[str],
1064 |     context_before: int,
1065 |     context_after: int,
1066 |     min_sentence_chars: int,
1067 |     max_sentence_chars: int,
1068 |     min_edit_similarity: float,
1069 |     max_edit_similarity: float,
1070 | ) -> pd.DataFrame:
1071 |     selected_keys = reservoir_sample_article_keys(
1072 |         connection,
1073 |         schema,
1074 |         max_articles=max_articles,
1075 |         seed=sampling_seed,
1076 |         sources=sources,
1077 |     )
1078 | 
1079 |     records: list[dict[str, Any]] = []
1080 |     processed_articles = 0
1081 | 
1082 |     for key, versions in load_selected_versions(connection, schema, selected_keys):
1083 |         processed_articles += 1
1084 |         article_records = extract_episodes_from_article(
1085 |             key,
1086 |             versions,
1087 |             context_before=context_before,
1088 |             context_after=context_after,
1089 |             min_sentence_chars=min_sentence_chars,
1090 |             max_sentence_chars=max_sentence_chars,
1091 |             min_edit_similarity=min_edit_similarity,
1092 |             max_edit_similarity=max_edit_similarity,
1093 |         )
1094 |         records.extend(article_records)
1095 | 
1096 |         if processed_articles % 1000 == 0:
1097 |             print(
1098 |                 f"Processed articles={processed_articles:,}; "
1099 |                 f"episodes={len(records):,}"
1100 |             )
1101 | 
1102 |         if max_episodes > 0 and len(records) >= max_episodes:
1103 |             records = records[:max_episodes]
1104 |             break
1105 | 
1106 |     if not records:
1107 |         raise ValueError(
1108 |             "No revision-lineage episodes were extracted. Run --inspect-only, "
1109 |             "check the article schema, or relax the edit/sentence filters."
1110 |         )
1111 | 
1112 |     frame = pd.DataFrame.from_records(records)
1113 |     frame = frame.drop_duplicates(subset=["episode_id"]).reset_index(drop=True)
1114 |     return frame
1115 | 
1116 | 
1117 | def save_episode_cache(frame: pd.DataFrame, path: Path) -> None:
1118 |     path.parent.mkdir(parents=True, exist_ok=True)
1119 |     suffixes = "".join(path.suffixes).lower()
1120 | 
1121 |     if suffixes.endswith(".parquet"):
1122 |         try:
1123 |             frame.to_parquet(path, index=False)
1124 |         except ImportError as exc:
1125 |             raise SystemExit(
1126 |                 "Saving Parquet requires pyarrow. Install it or use .csv.gz."
1127 |             ) from exc
1128 |     else:
1129 |         compression = "gzip" if suffixes.endswith(".gz") else None
1130 |         frame.to_csv(path, index=False, compression=compression)
1131 | 
1132 | 
1133 | def load_episode_cache(path: Path) -> pd.DataFrame:
1134 |     suffixes = "".join(path.suffixes).lower()
1135 |     if suffixes.endswith(".parquet"):
1136 |         return pd.read_parquet(path)
1137 |     return pd.read_csv(path)
1138 | 
1139 | 
1140 | # ---------------------------------------------------------------------------
1141 | # Mechanism-ablation feature sets and model
1142 | # ---------------------------------------------------------------------------
1143 | 
1144 | 
1145 | BASE_TEXT_COLUMNS = [
1146 |     "context_before",
1147 |     "retained_sentence",
1148 |     "context_after",
1149 | ]
1150 | BASE_NUMERIC_COLUMNS = [
1151 |     "version_index",
1152 |     "n_versions",
1153 |     "sentence_position",
1154 |     "retained_chars",
1155 |     "retained_tokens",
1156 | ]
1157 | BASE_CATEGORICAL_COLUMNS = ["source"]
1158 | 
1159 | REJECTED_TEXT_COLUMNS = ["rejected_sentence"]
1160 | EDIT_GEOMETRY_COLUMNS = [
1161 |     "rejected_chars",
1162 |     "rejected_tokens",
1163 |     "char_delta",
1164 |     "token_delta",
1165 | ]
1166 | LEXICAL_RELATION_COLUMNS = [
1167 |     "edit_similarity",
1168 |     "lexical_jaccard",
1169 | ]
1170 | ALL_TEXT_COLUMNS = BASE_TEXT_COLUMNS + REJECTED_TEXT_COLUMNS
1171 | ALL_PREFERENCE_NUMERIC_COLUMNS = (
1172 |     EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
1173 | )
1174 | 
1175 | 
1176 | @dataclasses.dataclass(frozen=True)
1177 | class FeatureSpec:
1178 |     name: str
1179 |     text_columns: tuple[str, ...]
1180 |     numeric_columns: tuple[str, ...]
1181 |     categorical_columns: tuple[str, ...]
1182 |     description: str
1183 |     uses_matched_shuffle: bool = False
1184 | 
1185 | 
1186 | def make_feature_specs(profile: str) -> list[FeatureSpec]:
1187 |     """Return coherent, baseline-conditioned mechanism ablations.
1188 | 
1189 |     Every mechanism model includes the complete current-text baseline. This
1190 |     makes each gain interpretable as incremental information beyond the
1191 |     retained sentence and article context.
1192 |     """
1193 |     base_text = tuple(BASE_TEXT_COLUMNS)
1194 |     base_num = tuple(BASE_NUMERIC_COLUMNS)
1195 |     base_cat = tuple(BASE_CATEGORICAL_COLUMNS)
1196 | 
1197 |     def spec(
1198 |         name: str,
1199 |         *,
1200 |         text: Sequence[str] = (),
1201 |         numeric: Sequence[str] = (),
1202 |         description: str,
1203 |         shuffled: bool = False,
1204 |     ) -> FeatureSpec:
1205 |         return FeatureSpec(
1206 |             name=name,
1207 |             text_columns=base_text + tuple(text),
1208 |             numeric_columns=base_num + tuple(numeric),
1209 |             categorical_columns=base_cat,
1210 |             description=description,
1211 |             uses_matched_shuffle=shuffled,
1212 |         )
1213 | 
1214 |     specs = [
1215 |         spec(
1216 |             "baseline",
1217 |             description="Current context, retained sentence and metadata.",
1218 |         ),
1219 |         spec(
1220 |             "plus_edit_geometry",
1221 |             numeric=EDIT_GEOMETRY_COLUMNS,
1222 |             description=(
1223 |                 "Baseline plus rejected length and edit-size geometry."
1224 |             ),
1225 |         ),
1226 |         spec(
1227 |             "plus_lexical_relation",
1228 |             numeric=LEXICAL_RELATION_COLUMNS,
1229 |             description=(
1230 |                 "Baseline plus scalar similarity/overlap between the pair."
1231 |             ),
1232 |         ),
1233 |         spec(
1234 |             "plus_rejected_text",
1235 |             text=REJECTED_TEXT_COLUMNS,
1236 |             description="Baseline plus the authentic rejected words.",
1237 |         ),
1238 |         spec(
1239 |             "plus_text_geometry",
1240 |             text=REJECTED_TEXT_COLUMNS,
1241 |             numeric=EDIT_GEOMETRY_COLUMNS,
1242 |             description="Baseline plus rejected text and edit geometry.",
1243 |         ),
1244 |     ]
1245 | 
1246 |     if profile == "full":
1247 |         specs.extend(
1248 |             [
1249 |                 spec(
1250 |                     "plus_text_lexical",
1251 |                     text=REJECTED_TEXT_COLUMNS,
1252 |                     numeric=LEXICAL_RELATION_COLUMNS,
1253 |                     description=(
1254 |                         "Baseline plus rejected text and lexical relation."
1255 |                     ),
1256 |                 ),
1257 |                 spec(
1258 |                     "plus_geometry_lexical",
1259 |                     numeric=(
1260 |                         EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
1261 |                     ),
1262 |                     description=(
1263 |                         "Baseline plus all non-text preference evidence."
1264 |                     ),
1265 |                 ),
1266 |             ]
1267 |         )
1268 | 
1269 |     specs.extend(
1270 |         [
1271 |             spec(
1272 |                 "full_preference",
1273 |                 text=REJECTED_TEXT_COLUMNS,
1274 |                 numeric=(
1275 |                     EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
1276 |                 ),
1277 |                 description=(
1278 |                     "Baseline plus authentic rejected text, geometry and "
1279 |                     "lexical relation."
1280 |                 ),
1281 |             ),
1282 |             spec(
1283 |                 "matched_shuffled_full_preference",
1284 |                 text=REJECTED_TEXT_COLUMNS,
1285 |                 numeric=(
1286 |                     EDIT_GEOMETRY_COLUMNS + LEXICAL_RELATION_COLUMNS
1287 |                 ),
1288 |                 description=(
1289 |                     "Full bundle after a local matched shuffle of rejected "
1290 |                     "text, with every pair-derived metric recomputed."
1291 |                 ),
1292 |                 shuffled=True,
1293 |             ),
1294 |         ]
1295 |     )
1296 |     return specs
1297 | 
1298 | 
1299 | def build_model_pipeline(
1300 |     *,
1301 |     feature_spec: FeatureSpec,
1302 |     seed: int,
1303 |     tfidf_max_features: int,
1304 |     tfidf_min_df: int,
1305 |     logistic_c: float,
1306 |     logistic_solver: str,
1307 |     logistic_max_iter: int,
1308 | ):
1309 |     from sklearn.compose import ColumnTransformer
1310 |     from sklearn.feature_extraction.text import TfidfVectorizer
1311 |     from sklearn.impute import SimpleImputer
1312 |     from sklearn.linear_model import LogisticRegression
1313 |     from sklearn.pipeline import Pipeline
1314 |     from sklearn.preprocessing import OneHotEncoder, StandardScaler
1315 | 
1316 |     transformers: list[tuple[str, Any, Any]] = []
1317 | 
1318 |     for column in feature_spec.text_columns:
1319 |         transformers.append(
1320 |             (
1321 |                 f"text_{column}",
1322 |                 TfidfVectorizer(
1323 |                     lowercase=True,
1324 |                     ngram_range=(1, 2),
1325 |                     min_df=tfidf_min_df,
1326 |                     max_df=1.0,
1327 |                     max_features=tfidf_max_features,
1328 |                     sublinear_tf=True,
1329 |                     norm="l2",
1330 |                 ),
1331 |                 column,
1332 |             )
1333 |         )
1334 | 
1335 |     if feature_spec.numeric_columns:
1336 |         numeric_pipe = Pipeline(
1337 |             [
1338 |                 ("imputer", SimpleImputer(strategy="median")),
1339 |                 ("scale", StandardScaler(with_mean=False)),
1340 |             ]
1341 |         )
1342 |         transformers.append(
1343 |             ("numeric", numeric_pipe, list(feature_spec.numeric_columns))
1344 |         )
1345 | 
1346 |     if feature_spec.categorical_columns:
1347 |         transformers.append(
1348 |             (
1349 |                 "categorical",
1350 |                 OneHotEncoder(handle_unknown="ignore"),
1351 |                 list(feature_spec.categorical_columns),
1352 |             )
1353 |         )
1354 | 
1355 |     preprocessor = ColumnTransformer(
1356 |         transformers=transformers,
1357 |         remainder="drop",
1358 |         sparse_threshold=0.1,
1359 |     )
1360 | 
1361 |     classifier = LogisticRegression(
1362 |         C=logistic_c,
1363 |         l1_ratio=0.0,
1364 |         solver=logistic_solver,
1365 |         max_iter=logistic_max_iter,
1366 |         tol=1e-4,
1367 |         class_weight=None,
1368 |         random_state=seed,
1369 |     )
1370 | 
1371 |     return Pipeline(
1372 |         [("preprocessor", preprocessor), ("classifier", classifier)]
1373 |     )
1374 | 
1375 | 
1376 | def fit_and_score(
1377 |     train_df: pd.DataFrame,
1378 |     test_df: pd.DataFrame,
1379 |     *,
1380 |     feature_spec: FeatureSpec,
1381 |     target_column: str,
1382 |     seed: int,
1383 |     tfidf_max_features: int,
1384 |     tfidf_min_df: int,
1385 |     logistic_c: float,
1386 |     logistic_solver: str,
1387 |     logistic_max_iter: int,
1388 | ) -> tuple[np.ndarray, dict[str, Any]]:
1389 |     from sklearn.metrics import (
1390 |         accuracy_score,
1391 |         average_precision_score,
1392 |         brier_score_loss,
1393 |         log_loss,
1394 |         roc_auc_score,
1395 |     )
1396 | 
1397 |     train_model_df = train_df.copy()
1398 |     test_model_df = test_df.copy()
1399 |     for column in set(feature_spec.text_columns):
1400 |         train_model_df[column] = (
1401 |             train_model_df[column]
1402 |             .fillna("")
1403 |             .astype(str)
1404 |             .map(normalise_space)
1405 |             .replace("", "__EMPTY__")
1406 |         )
1407 |         test_model_df[column] = (
1408 |             test_model_df[column]
1409 |             .fillna("")
1410 |             .astype(str)
1411 |             .map(normalise_space)
1412 |             .replace("", "__EMPTY__")
1413 |         )
1414 | 
1415 |     y_train = train_model_df[target_column].astype(int).to_numpy()
1416 |     y_test = test_model_df[target_column].astype(int).to_numpy()
1417 |     if np.unique(y_train).size < 2:
1418 |         raise ValueError("The training split contains only one target class.")
1419 | 
1420 |     pipeline = build_model_pipeline(
1421 |         feature_spec=feature_spec,
1422 |         seed=seed,
1423 |         tfidf_max_features=tfidf_max_features,
1424 |         tfidf_min_df=tfidf_min_df,
1425 |         logistic_c=logistic_c,
1426 |         logistic_solver=logistic_solver,
1427 |         logistic_max_iter=logistic_max_iter,
1428 |     )
1429 |     pipeline.fit(train_model_df, y_train)
1430 | 
1431 |     probabilities = pipeline.predict_proba(test_model_df)[:, 1]
1432 |     predictions = (probabilities >= 0.5).astype(int)
1433 | 
1434 |     train_prevalence = float(y_train.mean())
1435 |     test_prevalence = float(y_test.mean())
1436 |     null_probabilities = np.full(
1437 |         len(y_test), np.clip(train_prevalence, 1e-12, 1.0 - 1e-12)
1438 |     )
1439 |     iterations = int(np.max(pipeline.named_steps["classifier"].n_iter_))
1440 | 
1441 |     metrics = {
1442 |         "loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
1443 |         "brier": float(brier_score_loss(y_test, probabilities)),
1444 |         "auc": float(roc_auc_score(y_test, probabilities)),
1445 |         "average_precision": float(
1446 |             average_precision_score(y_test, probabilities)
1447 |         ),
1448 |         "accuracy": float(accuracy_score(y_test, predictions)),
1449 |         "train_prevalence": train_prevalence,
1450 |         "test_prevalence": test_prevalence,
1451 |         "mean_predicted_probability": float(probabilities.mean()),
1452 |         "probability_min": float(np.min(probabilities)),
1453 |         "probability_p01": float(np.quantile(probabilities, 0.01)),
1454 |         "probability_p05": float(np.quantile(probabilities, 0.05)),
1455 |         "probability_median": float(np.median(probabilities)),
1456 |         "probability_p95": float(np.quantile(probabilities, 0.95)),
1457 |         "probability_p99": float(np.quantile(probabilities, 0.99)),
1458 |         "probability_max": float(np.max(probabilities)),
1459 |         "calibration_gap": float(probabilities.mean() - test_prevalence),
1460 |         "solver": logistic_solver,
1461 |         "converged": bool(iterations < logistic_max_iter),
1462 |         "n_iter": iterations,
1463 |         "null_log_loss": float(
1464 |             log_loss(y_test, null_probabilities, labels=[0, 1])
1465 |         ),
1466 |         "null_brier": float(brier_score_loss(y_test, null_probabilities)),
1467 |     }
1468 |     return probabilities, metrics
1469 | 
1470 | 
1471 | def _local_matched_donor_indices(
1472 |     frame: pd.DataFrame,
1473 |     *,
1474 |     seed: int,
1475 |     block_size: int,
1476 | ) -> np.ndarray:
1477 |     """Create a no-self local permutation in structural-neighbour blocks.
1478 | 
1479 |     Rows are sorted by source and edit geometry before being partitioned into
1480 |     small blocks. Each block receives a non-zero cyclic shift. This is not an
1481 |     exact conditional randomisation test, but it produces a substantially
1482 |     harder and more coherent semantic control than an unrestricted shuffle.
1483 |     """
1484 |     if block_size < 2:
1485 |         raise ValueError("matched shuffle block size must be at least 2")
1486 | 
1487 |     rng = np.random.default_rng(seed)
1488 |     donor = np.arange(len(frame), dtype=int)
1489 |     working = frame.reset_index(drop=False).rename(columns={"index": "_row"})
1490 | 
1491 |     sort_columns = [
1492 |         "source",
1493 |         "retained_chars",
1494 |         "rejected_chars",
1495 |         "edit_similarity",
1496 |         "lexical_jaccard",
1497 |         "version_index",
1498 |         "sentence_position",
1499 |     ]
1500 |     working = working.sort_values(sort_columns, kind="mergesort")
1501 | 
1502 |     for _source, source_part in working.groupby("source", sort=False):
1503 |         positions = source_part["_row"].to_numpy(dtype=int)
1504 |         if len(positions) <= 1:
1505 |             continue
1506 | 
1507 |         blocks = [
1508 |             positions[start : start + block_size]
1509 |             for start in range(0, len(positions), block_size)
1510 |         ]
1511 |         if len(blocks) > 1 and len(blocks[-1]) == 1:
1512 |             blocks[-2] = np.concatenate([blocks[-2], blocks[-1]])
1513 |             blocks.pop()
1514 | 
1515 |         for block in blocks:
1516 |             if len(block) <= 1:
1517 |                 continue
1518 |             shift = int(rng.integers(1, len(block)))
1519 |             donor[block] = np.roll(block, shift)
1520 | 
1521 |     return donor
1522 | 
1523 | 
1524 | def matched_shuffle_rejected_text(
1525 |     frame: pd.DataFrame,
1526 |     *,
1527 |     seed: int,
1528 |     block_size: int,
1529 | ) -> tuple[pd.DataFrame, dict[str, float]]:
1530 |     """Shuffle rejected text locally and recompute every derived pair metric."""
1531 |     shuffled = frame.reset_index(drop=True).copy()
1532 |     donor = _local_matched_donor_indices(
1533 |         shuffled, seed=seed, block_size=block_size
1534 |     )
1535 | 
1536 |     original_rejected = shuffled["rejected_sentence"].fillna("").astype(str)
1537 |     donor_rejected = original_rejected.iloc[donor].reset_index(drop=True)
1538 |     shuffled["rejected_sentence"] = donor_rejected
1539 | 
1540 |     shuffled["rejected_chars"] = donor_rejected.str.len().astype(int)
1541 |     shuffled["rejected_tokens"] = donor_rejected.map(
1542 |         lambda value: len(tokenise(value))
1543 |     ).astype(int)
1544 |     shuffled["char_delta"] = (
1545 |         shuffled["retained_chars"].astype(int)
1546 |         - shuffled["rejected_chars"].astype(int)
1547 |     )
1548 |     shuffled["token_delta"] = (
1549 |         shuffled["retained_tokens"].astype(int)
1550 |         - shuffled["rejected_tokens"].astype(int)
1551 |     )
1552 |     shuffled["edit_similarity"] = [
1553 |         difflib.SequenceMatcher(
1554 |             a=normalise_sentence(rejected),
1555 |             b=normalise_sentence(retained),
1556 |             autojunk=False,
1557 |         ).ratio()
1558 |         for rejected, retained in zip(
1559 |             shuffled["rejected_sentence"], shuffled["retained_sentence"]
1560 |         )
1561 |     ]
1562 |     shuffled["lexical_jaccard"] = [
1563 |         lexical_jaccard(rejected, retained)
1564 |         for rejected, retained in zip(
1565 |             shuffled["rejected_sentence"], shuffled["retained_sentence"]
1566 |         )
1567 |     ]
1568 | 
1569 |     same_text = np.asarray(
1570 |         [
1571 |             normalise_sentence(a) == normalise_sentence(b)
1572 |             for a, b in zip(original_rejected, donor_rejected)
1573 |         ],
1574 |         dtype=float,
1575 |     )
1576 |     diagnostics = {
1577 |         "same_rejected_text_rate": float(same_text.mean()),
1578 |         "mean_abs_rejected_chars_change": float(
1579 |             np.mean(
1580 |                 np.abs(
1581 |                     frame.reset_index(drop=True)["rejected_chars"].to_numpy()
1582 |                     - shuffled["rejected_chars"].to_numpy()
1583 |                 )
1584 |             )
1585 |         ),
1586 |         "mean_abs_similarity_change": float(
1587 |             np.mean(
1588 |                 np.abs(
1589 |                     frame.reset_index(drop=True)["edit_similarity"].to_numpy()
1590 |                     - shuffled["edit_similarity"].to_numpy()
1591 |                 )
1592 |             )
1593 |         ),
1594 |     }
1595 |     return shuffled, diagnostics
1596 | 
1597 | 
1598 | def build_group_loss_stats(
1599 |     test_df: pd.DataFrame,
1600 |     *,
1601 |     target_column: str,
1602 |     seed: int,
1603 |     predictions: dict[str, np.ndarray],
1604 | ) -> pd.DataFrame:
1605 |     y_true = test_df[target_column].astype(int).to_numpy()
1606 |     row_stats = pd.DataFrame(
1607 |         {
1608 |             "group_id": test_df["article_key"].astype(str).to_numpy(),
1609 |             "n_rows": 1,
1610 |         },
1611 |         index=test_df.index,
1612 |     )
1613 |     for name, probability in predictions.items():
1614 |         row_stats[f"{name}__log_sum"] = log_loss_components(
1615 |             y_true, probability
1616 |         )
1617 |         row_stats[f"{name}__brier_sum"] = brier_components(
1618 |             y_true, probability
1619 |         )
1620 |     grouped = row_stats.groupby("group_id", as_index=False).sum(
1621 |         numeric_only=True
1622 |     )
1623 |     grouped["seed"] = seed
1624 |     return grouped
1625 | 
1626 | 
1627 | def evaluate_seed(
1628 |     frame: pd.DataFrame,
1629 |     *,
1630 |     seed: int,
1631 |     test_fraction: float,
1632 |     target_column: str,
1633 |     tfidf_max_features: int,
1634 |     tfidf_min_df: int,
1635 |     logistic_c: float,
1636 |     logistic_solver: str,
1637 |     logistic_max_iter: int,
1638 |     ablation_profile: str,
1639 |     matched_shuffle_block_size: int,
1640 | ) -> EvaluationBundle:
1641 |     rng = np.random.default_rng(seed)
1642 |     groups = np.asarray(
1643 |         sorted(frame["article_key"].dropna().astype(str).unique())
1644 |     )
1645 |     if len(groups) < 2:
1646 |         raise ValueError("At least two article groups are required.")
1647 |     rng.shuffle(groups)
1648 |     n_test = max(1, int(round(len(groups) * test_fraction)))
1649 |     n_test = min(n_test, len(groups) - 1)
1650 |     test_groups = set(groups[:n_test])
1651 | 
1652 |     group_values = frame["article_key"].astype(str)
1653 |     train_df = frame[~group_values.isin(test_groups)].copy().reset_index(drop=True)
1654 |     test_df = frame[group_values.isin(test_groups)].copy().reset_index(drop=True)
1655 | 
1656 |     shuffled_train, train_shuffle_diagnostics = matched_shuffle_rejected_text(
1657 |         train_df,
1658 |         seed=seed + 10_000,
1659 |         block_size=matched_shuffle_block_size,
1660 |     )
1661 |     shuffled_test, test_shuffle_diagnostics = matched_shuffle_rejected_text(
1662 |         test_df,
1663 |         seed=seed + 20_000,
1664 |         block_size=matched_shuffle_block_size,
1665 |     )
1666 |     print(
1667 |         "Matched-shuffle diagnostics: "
1668 |         f"train_same={train_shuffle_diagnostics['same_rejected_text_rate']:.4f}; "
1669 |         f"test_same={test_shuffle_diagnostics['same_rejected_text_rate']:.4f}; "
1670 |         f"test_mean_abs_similarity_change="
1671 |         f"{test_shuffle_diagnostics['mean_abs_similarity_change']:.4f}"
1672 |     )
1673 | 
1674 |     rows: list[ResultRow] = []
1675 |     predictions: dict[str, np.ndarray] = {}
1676 |     for feature_spec in make_feature_specs(ablation_profile):
1677 |         model_train = shuffled_train if feature_spec.uses_matched_shuffle else train_df
1678 |         model_test = shuffled_test if feature_spec.uses_matched_shuffle else test_df
1679 |         probability, metrics = fit_and_score(
1680 |             model_train,
1681 |             model_test,
1682 |             feature_spec=feature_spec,
1683 |             target_column=target_column,
1684 |             seed=seed,
1685 |             tfidf_max_features=tfidf_max_features,
1686 |             tfidf_min_df=tfidf_min_df,
1687 |             logistic_c=logistic_c,
1688 |             logistic_solver=logistic_solver,
1689 |             logistic_max_iter=logistic_max_iter,
1690 |         )
1691 |         predictions[feature_spec.name] = probability
1692 |         rows.append(
1693 |             ResultRow(
1694 |                 track="newsedits",
1695 |                 condition="sentence_revision_mechanism_ablation",
1696 |                 seed=seed,
1697 |                 target=target_column,
1698 |                 feature_set=feature_spec.name,
1699 |                 n_train=len(model_train),
1700 |                 n_test=len(model_test),
1701 |                 n_train_groups=model_train["article_key"].nunique(),
1702 |                 n_test_groups=model_test["article_key"].nunique(),
1703 |                 **metrics,
1704 |             )
1705 |         )
1706 | 
1707 |     stats = build_group_loss_stats(
1708 |         test_df,
1709 |         target_column=target_column,
1710 |         seed=seed,
1711 |         predictions=predictions,
1712 |     )
1713 |     return EvaluationBundle(rows=rows, group_loss_stats=stats)
1714 | 
1715 | 
1716 | # ---------------------------------------------------------------------------
1717 | # Generic paired comparisons and hierarchical bootstrap
1718 | # ---------------------------------------------------------------------------
1719 | 
1720 | 
1721 | def comparison_definitions(available: set[str]) -> list[tuple[str, str, str]]:
1722 |     requested = [
1723 |         ("full_vs_baseline", "baseline", "full_preference"),
1724 |         ("geometry_vs_baseline", "baseline", "plus_edit_geometry"),
1725 |         ("lexical_vs_baseline", "baseline", "plus_lexical_relation"),
1726 |         ("rejected_text_vs_baseline", "baseline", "plus_rejected_text"),
1727 |         ("text_geometry_vs_baseline", "baseline", "plus_text_geometry"),
1728 |         ("text_lexical_vs_baseline", "baseline", "plus_text_lexical"),
1729 |         (
1730 |             "geometry_lexical_vs_baseline",
1731 |             "baseline",
1732 |             "plus_geometry_lexical",
1733 |         ),
1734 |         (
1735 |             "semantic_increment_beyond_nontext",
1736 |             "plus_geometry_lexical",
1737 |             "full_preference",
1738 |         ),
1739 |         (
1740 |             "geometry_increment_beyond_text_lexical",
1741 |             "plus_text_lexical",
1742 |             "full_preference",
1743 |         ),
1744 |         (
1745 |             "lexical_increment_beyond_text_geometry",
1746 |             "plus_text_geometry",
1747 |             "full_preference",
1748 |         ),
1749 |         (
1750 |             "authentic_vs_matched_shuffle",
1751 |             "matched_shuffled_full_preference",
1752 |             "full_preference",
1753 |         ),
1754 |     ]
1755 |     return [item for item in requested if item[1] in available and item[2] in available]
1756 | 
1757 | 
1758 | def hierarchical_bootstrap_comparison(
1759 |     stats: pd.DataFrame,
1760 |     *,
1761 |     reference_feature_set: str,
1762 |     candidate_feature_set: str,
1763 |     metric: str,
1764 |     samples: int,
1765 |     confidence_level: float,
1766 |     seed: int,
1767 | ) -> tuple[float, float]:
1768 |     if samples <= 0:
1769 |         return float("nan"), float("nan")
1770 |     if metric not in {"log_loss", "brier"}:
1771 |         raise ValueError(metric)
1772 | 
1773 |     suffix = "log_sum" if metric == "log_loss" else "brier_sum"
1774 |     reference_column = f"{reference_feature_set}__{suffix}"
1775 |     candidate_column = f"{candidate_feature_set}__{suffix}"
1776 |     columns = ["n_rows", reference_column, candidate_column]
1777 | 
1778 |     rng = np.random.default_rng(seed)
1779 |     seed_values = np.asarray(sorted(stats["seed"].unique()), dtype=int)
1780 |     by_seed = {
1781 |         int(seed_value): stats.loc[
1782 |             stats["seed"] == seed_value, columns
1783 |         ].to_numpy(dtype=float)
1784 |         for seed_value in seed_values
1785 |     }
1786 | 
1787 |     draws = np.empty(samples, dtype=float)
1788 |     for draw_index in range(samples):
1789 |         sampled_seeds = rng.choice(
1790 |             seed_values, size=len(seed_values), replace=True
1791 |         )
1792 |         totals = np.zeros(3, dtype=float)
1793 |         for sampled_seed in sampled_seeds:
1794 |             array = by_seed[int(sampled_seed)]
1795 |             indices = rng.integers(0, len(array), size=len(array))
1796 |             totals += array[indices].sum(axis=0)
1797 |         draws[draw_index] = (totals[1] - totals[2]) / totals[0]
1798 | 
1799 |     alpha = 1.0 - confidence_level
1800 |     return (
1801 |         float(np.quantile(draws, alpha / 2.0)),
1802 |         float(np.quantile(draws, 1.0 - alpha / 2.0)),
1803 |     )
1804 | 
1805 | 
1806 | def build_summary_rows(
1807 |     rows: list[ResultRow],
1808 |     group_stats: pd.DataFrame,
1809 |     *,
1810 |     bootstrap_samples: int,
1811 |     confidence_level: float,
1812 |     bootstrap_seed: int,
1813 | ) -> list[SummaryRow]:
1814 |     frame = pd.DataFrame([dataclasses.asdict(row) for row in rows])
1815 |     available = set(frame["feature_set"].unique())
1816 |     summaries: list[SummaryRow] = []
1817 | 
1818 |     for comparison, reference, candidate in comparison_definitions(available):
1819 |         for metric, result_column in [("log_loss", "loss"), ("brier", "brier")]:
1820 |             per_seed: list[float] = []
1821 |             for seed_value in sorted(frame["seed"].unique()):
1822 |                 indexed = frame[frame["seed"] == seed_value].set_index(
1823 |                     "feature_set"
1824 |                 )
1825 |                 per_seed.append(
1826 |                     float(
1827 |                         indexed.loc[reference, result_column]
1828 |                         - indexed.loc[candidate, result_column]
1829 |                     )
1830 |                 )
1831 |             values = np.asarray(per_seed, dtype=float)
1832 |             low, high = hierarchical_bootstrap_comparison(
1833 |                 group_stats,
1834 |                 reference_feature_set=reference,
1835 |                 candidate_feature_set=candidate,
1836 |                 metric=metric,
1837 |                 samples=bootstrap_samples,
1838 |                 confidence_level=confidence_level,
1839 |                 seed=(
1840 |                     bootstrap_seed
1841 |                     + sum(ord(char) for char in comparison + metric)
1842 |                 ),
1843 |             )
1844 |             summaries.append(
1845 |                 SummaryRow(
1846 |                     track="newsedits",
1847 |                     condition="sentence_revision_mechanism_ablation",
1848 |                     comparison=comparison,
1849 |                     metric=metric,
1850 |                     reference_feature_set=reference,
1851 |                     candidate_feature_set=candidate,
1852 |                     n_seeds=len(values),
1853 |                     mean_gain=float(values.mean()),
1854 |                     seed_std=(
1855 |                         float(values.std(ddof=1)) if len(values) > 1 else 0.0
1856 |                     ),
1857 |                     ci_low=low,
1858 |                     ci_high=high,
1859 |                     positive_seeds=int((values > 0).sum()),
1860 |                     confidence_level=confidence_level,
1861 |                     bootstrap_samples=bootstrap_samples,
1862 |                 )
1863 |             )
1864 |     return summaries
1865 | 
1866 | 
1867 | # ---------------------------------------------------------------------------
1868 | # Reporting
1869 | # ---------------------------------------------------------------------------
1870 | 
1871 | 
1872 | def audit_episodes(frame: pd.DataFrame) -> None:
1873 |     print_header("NewsEdits revision-lineage audit")
1874 |     print(f"Episodes: {len(frame):,}")
1875 |     print(f"Articles: {frame['article_key'].nunique():,}")
1876 |     print(f"Sources: {frame['source'].nunique():,}")
1877 |     print(
1878 |         "Target revised-again rate: "
1879 |         f"{frame['revised_again_next_version'].mean():.6f}"
1880 |     )
1881 |     print("\nTarget counts:")
1882 |     print(
1883 |         frame["revised_again_next_version"]
1884 |         .value_counts(dropna=False)
1885 |         .sort_index()
1886 |         .to_string()
1887 |     )
1888 |     print("\nLargest sources:")
1889 |     print(frame["source"].value_counts().head(20).to_string())
1890 |     print("\nEdit similarity:")
1891 |     print(frame["edit_similarity"].describe().to_string())
1892 | 
1893 | 
1894 | def print_results(rows: list[ResultRow], summaries: list[SummaryRow]) -> None:
1895 |     result_frame = pd.DataFrame([dataclasses.asdict(row) for row in rows])
1896 |     aggregate = result_frame.groupby("feature_set")[[
1897 |         "loss",
1898 |         "brier",
1899 |         "auc",
1900 |         "average_precision",
1901 |         "accuracy",
1902 |         "mean_predicted_probability",
1903 |         "calibration_gap",
1904 |         "probability_p01",
1905 |         "probability_p99",
1906 |         "n_iter",
1907 |         "converged",
1908 |         "null_log_loss",
1909 |     ]].agg(["mean", "std"])
1910 | 
1911 |     print_header("Mechanism-ablation metrics across seeds")
1912 |     print(aggregate.to_string(float_format=lambda value: f"{value:.6f}"))
1913 | 
1914 |     sanity = result_frame.groupby("feature_set")[[
1915 |         "loss", "null_log_loss", "brier", "null_brier"
1916 |     ]].mean()
1917 |     sanity["log_loss_gain_vs_null"] = (
1918 |         sanity["null_log_loss"] - sanity["loss"]
1919 |     )
1920 |     sanity["brier_gain_vs_null"] = sanity["null_brier"] - sanity["brier"]
1921 |     print_header("Probability sanity check")
1922 |     print(sanity.to_string(float_format=lambda value: f"{value:.6f}"))
1923 | 
1924 |     summary_frame = pd.DataFrame(
1925 |         [dataclasses.asdict(row) for row in summaries]
1926 |     )
1927 |     print_header("Mechanism comparison summary")
1928 |     print(
1929 |         summary_frame[[
1930 |             "comparison",
1931 |             "metric",
1932 |             "reference_feature_set",
1933 |             "candidate_feature_set",
1934 |             "n_seeds",
1935 |             "mean_gain",
1936 |             "seed_std",
1937 |             "ci_low",
1938 |             "ci_high",
1939 |             "positive_seeds",
1940 |         ]].to_string(index=False, float_format=lambda value: f"{value:.6f}")
1941 |     )
1942 | 
1943 | # ---------------------------------------------------------------------------
1944 | # CLI
1945 | # ---------------------------------------------------------------------------
1946 | 
1947 | 
1948 | def parse_args(argv: Sequence[str]) -> argparse.Namespace:
1949 |     parser = argparse.ArgumentParser(
1950 |         description=(
1951 |             "Ablate semantic, geometric and lexical mechanisms in NewsEdits "
1952 |             "preference-future forecasting."
1953 |         )
1954 |     )
1955 |     parser.add_argument("--db", help="Path to the NewsEdits SQLite database.")
1956 |     parser.add_argument(
1957 |         "--articles-table",
1958 |         default=None,
1959 |         help="Optional article-version table name; otherwise auto-discovered.",
1960 |     )
1961 |     parser.add_argument(
1962 |         "--split-table",
1963 |         default=None,
1964 |         help=(
1965 |             "Optional official split-sentence table name. "
1966 |             "Defaults to split_sentences when present."
1967 |         ),
1968 |     )
1969 |     parser.add_argument(
1970 |         "--source-name",
1971 |         default=None,
1972 |         help=(
1973 |             "Source/outlet label for source-specific official databases. "
1974 |             "Defaults to a cleaned database filename."
1975 |         ),
1976 |     )
1977 |     parser.add_argument(
1978 |         "--inspect-only",
1979 |         action="store_true",
1980 |         help="Print SQLite tables/schema and exit.",
1981 |     )
1982 |     parser.add_argument(
1983 |         "--episode-cache",
1984 |         default=None,
1985 |         help=(
1986 |             "CSV, CSV.GZ or Parquet path. Existing cache is loaded unless "
1987 |             "--rebuild-cache is supplied."
1988 |         ),
1989 |     )
1990 |     parser.add_argument(
1991 |         "--rebuild-cache",
1992 |         action="store_true",
1993 |     )
1994 |     parser.add_argument(
1995 |         "--max-articles",
1996 |         type=int,
1997 |         default=10_000,
1998 |         help="Reservoir-sampled articles with 3+ versions. Use 0 for all.",
1999 |     )
2000 |     parser.add_argument(
2001 |         "--max-episodes",
2002 |         type=int,
2003 |         default=0,
2004 |         help="Stop after this many episodes; 0 means no episode cap.",
2005 |     )
2006 |     parser.add_argument(
2007 |         "--sampling-seed",
2008 |         type=int,
2009 |         default=1729,
2010 |     )
2011 |     parser.add_argument(
2012 |         "--source",
2013 |         action="append",
2014 |         default=[],
2015 |         help="Optional source/outlet filter; may be repeated.",
2016 |     )
2017 |     parser.add_argument(
2018 |         "--context-before",
2019 |         type=int,
2020 |         default=2,
2021 |     )
2022 |     parser.add_argument(
2023 |         "--context-after",
2024 |         type=int,
2025 |         default=1,
2026 |     )
2027 |     parser.add_argument(
2028 |         "--min-sentence-chars",
2029 |         type=int,
2030 |         default=25,
2031 |     )
2032 |     parser.add_argument(
2033 |         "--max-sentence-chars",
2034 |         type=int,
2035 |         default=600,
2036 |     )
2037 |     parser.add_argument(
2038 |         "--min-edit-similarity",
2039 |         type=float,
2040 |         default=0.20,
2041 |     )
2042 |     parser.add_argument(
2043 |         "--max-edit-similarity",
2044 |         type=float,
2045 |         default=0.98,
2046 |     )
2047 |     parser.add_argument("--seed", type=int, default=7)
2048 |     parser.add_argument(
2049 |         "--seeds",
2050 |         default=None,
2051 |         help="Comma-separated split/model seeds.",
2052 |     )
2053 |     parser.add_argument(
2054 |         "--test-fraction",
2055 |         type=float,
2056 |         default=0.2,
2057 |     )
2058 |     parser.add_argument(
2059 |         "--tfidf-max-features",
2060 |         type=int,
2061 |         default=40_000,
2062 |         help="Maximum TF-IDF features per text field.",
2063 |     )
2064 |     parser.add_argument(
2065 |         "--tfidf-min-df",
2066 |         type=int,
2067 |         default=2,
2068 |         help=(
2069 |             "Minimum document frequency per TF-IDF field. Use 2 or more "
2070 |             "to avoid one-off features destabilising small samples."
2071 |         ),
2072 |     )
2073 |     parser.add_argument(
2074 |         "--logistic-c",
2075 |         type=float,
2076 |         default=0.1,
2077 |         help=(
2078 |             "Inverse L2 regularisation strength. Smaller values are more "
2079 |             "regularised and usually better calibrated on small samples."
2080 |         ),
2081 |     )
2082 |     parser.add_argument(
2083 |         "--logistic-solver",
2084 |         choices=["auto", "liblinear", "saga"],
2085 |         default="auto",
2086 |         help=(
2087 |             "Use liblinear for small/medium binary samples and saga for "
2088 |             "large sparse samples. auto switches at 50,000 training rows."
2089 |         ),
2090 |     )
2091 |     parser.add_argument(
2092 |         "--logistic-max-iter",
2093 |         type=int,
2094 |         default=10000,
2095 |         help="Maximum solver iterations; convergence is reported per model.",
2096 |     )
2097 |     parser.add_argument(
2098 |         "--ablation-profile",
2099 |         choices=["core", "full"],
2100 |         default="full",
2101 |         help=(
2102 |             "core runs the main mechanism models; full also runs the "
2103 |             "pairwise complementary ablations needed to estimate semantic, "
2104 |             "geometry and lexical increments."
2105 |         ),
2106 |     )
2107 |     parser.add_argument(
2108 |         "--matched-shuffle-block-size",
2109 |         type=int,
2110 |         default=32,
2111 |         help=(
2112 |             "Local structural-neighbour block size for the coherent rejected-"
2113 |             "text permutation control."
2114 |         ),
2115 |     )
2116 |     parser.add_argument(
2117 |         "--bootstrap-samples",
2118 |         type=int,
2119 |         default=2000,
2120 |     )
2121 |     parser.add_argument(
2122 |         "--bootstrap-seed",
2123 |         type=int,
2124 |         default=2718,
2125 |     )
2126 |     parser.add_argument(
2127 |         "--confidence-level",
2128 |         type=float,
2129 |         default=0.95,
2130 |     )
2131 |     parser.add_argument(
2132 |         "--out",
2133 |         default=None,
2134 |         help="Per-seed result CSV.",
2135 |     )
2136 |     parser.add_argument(
2137 |         "--summary-out",
2138 |         default=None,
2139 |         help="PFI summary CSV.",
2140 |     )
2141 |     return parser.parse_args(argv)
2142 | 
2143 | 
2144 | def main(argv: Sequence[str]) -> int:
2145 |     args = parse_args(argv)
2146 | 
2147 |     if not 0.0 < args.test_fraction < 1.0:
2148 |         raise SystemExit("--test-fraction must be between 0 and 1.")
2149 |     if not 0.0 <= args.min_edit_similarity < args.max_edit_similarity <= 1.0:
2150 |         raise SystemExit("Edit similarity bounds must satisfy 0 <= min < max <= 1.")
2151 |     if args.bootstrap_samples < 0:
2152 |         raise SystemExit("--bootstrap-samples must be non-negative.")
2153 |     if args.tfidf_min_df < 1:
2154 |         raise SystemExit("--tfidf-min-df must be at least 1.")
2155 |     if args.logistic_c <= 0:
2156 |         raise SystemExit("--logistic-c must be positive.")
2157 |     if args.logistic_max_iter < 1:
2158 |         raise SystemExit("--logistic-max-iter must be at least 1.")
2159 |     if args.matched_shuffle_block_size < 2:
2160 |         raise SystemExit("--matched-shuffle-block-size must be at least 2.")
2161 | 
2162 |     cache_path = Path(args.episode_cache) if args.episode_cache else None
2163 | 
2164 |     if cache_path and cache_path.exists() and not args.rebuild_cache:
2165 |         print(f"Loading episode cache: {cache_path}")
2166 |         episodes = load_episode_cache(cache_path)
2167 |     else:
2168 |         if not args.db:
2169 |             raise SystemExit(
2170 |                 "--db is required when an episode cache is not available."
2171 |             )
2172 | 
2173 |         db_path = Path(args.db)
2174 |         if not db_path.exists():
2175 |             raise SystemExit(f"SQLite database not found: {db_path}")
2176 | 
2177 |         connection = sqlite3.connect(str(db_path))
2178 |         try:
2179 |             if args.inspect_only:
2180 |                 inspect_database(
2181 |                     connection,
2182 |                     args.articles_table,
2183 |                     args.split_table,
2184 |                 )
2185 |                 return 0
2186 | 
2187 |             try:
2188 |                 split_schema = discover_split_sentence_schema(
2189 |                     connection,
2190 |                     args.split_table,
2191 |                 )
2192 |             except ValueError:
2193 |                 split_schema = None
2194 | 
2195 |             if split_schema is not None:
2196 |                 source_name = infer_source_name(
2197 |                     db_path,
2198 |                     args.source_name,
2199 |                 )
2200 |                 print_header("Detected official split-sentence schema")
2201 |                 print(dataclasses.asdict(split_schema))
2202 |                 print(f"Source label: {source_name}")
2203 | 
2204 |                 episodes = build_episode_dataframe_from_split_sentences(
2205 |                     connection,
2206 |                     split_schema,
2207 |                     source_name=source_name,
2208 |                     max_articles=args.max_articles,
2209 |                     max_episodes=args.max_episodes,
2210 |                     sampling_seed=args.sampling_seed,
2211 |                     context_before=args.context_before,
2212 |                     context_after=args.context_after,
2213 |                     min_sentence_chars=args.min_sentence_chars,
2214 |                     max_sentence_chars=args.max_sentence_chars,
2215 |                     min_edit_similarity=args.min_edit_similarity,
2216 |                     max_edit_similarity=args.max_edit_similarity,
2217 |                 )
2218 |             else:
2219 |                 schema = discover_article_schema(
2220 |                     connection,
2221 |                     args.articles_table,
2222 |                 )
2223 |                 print_header("Detected full-article schema")
2224 |                 print(dataclasses.asdict(schema))
2225 | 
2226 |                 episodes = build_episode_dataframe(
2227 |                     connection,
2228 |                     schema,
2229 |                     max_articles=args.max_articles,
2230 |                     max_episodes=args.max_episodes,
2231 |                     sampling_seed=args.sampling_seed,
2232 |                     sources=args.source,
2233 |                     context_before=args.context_before,
2234 |                     context_after=args.context_after,
2235 |                     min_sentence_chars=args.min_sentence_chars,
2236 |                     max_sentence_chars=args.max_sentence_chars,
2237 |                     min_edit_similarity=args.min_edit_similarity,
2238 |                     max_edit_similarity=args.max_edit_similarity,
2239 |                 )
2240 |         finally:
2241 |             connection.close()
2242 | 
2243 |         if cache_path:
2244 |             save_episode_cache(episodes, cache_path)
2245 |             print(f"Saved episode cache: {cache_path}")
2246 | 
2247 |     required_columns = {
2248 |         "article_key",
2249 |         "source",
2250 |         "context_before",
2251 |         "retained_sentence",
2252 |         "context_after",
2253 |         "rejected_sentence",
2254 |         "revised_again_next_version",
2255 |         *BASE_NUMERIC_COLUMNS,
2256 |         *ALL_PREFERENCE_NUMERIC_COLUMNS,
2257 |     }
2258 |     missing = sorted(required_columns - set(episodes.columns))
2259 |     if missing:
2260 |         raise SystemExit(
2261 |             f"Episode cache is missing required columns: {missing}"
2262 |         )
2263 | 
2264 |     audit_episodes(episodes)
2265 | 
2266 |     seeds = parse_int_list(args.seeds, args.seed)
2267 |     if args.logistic_solver == "auto":
2268 |         estimated_train_rows = int(round(len(episodes) * (1.0 - args.test_fraction)))
2269 |         logistic_solver = (
2270 |             "liblinear" if estimated_train_rows < 50_000 else "saga"
2271 |         )
2272 |     else:
2273 |         logistic_solver = args.logistic_solver
2274 |     print(f"Logistic solver: {logistic_solver}")
2275 | 
2276 |     all_rows: list[ResultRow] = []
2277 |     all_stats: list[pd.DataFrame] = []
2278 | 
2279 |     for seed in seeds:
2280 |         print(f"\nEvaluating seed {seed}...")
2281 |         bundle = evaluate_seed(
2282 |             episodes,
2283 |             seed=seed,
2284 |             test_fraction=args.test_fraction,
2285 |             target_column="revised_again_next_version",
2286 |             tfidf_max_features=args.tfidf_max_features,
2287 |             tfidf_min_df=args.tfidf_min_df,
2288 |             logistic_c=args.logistic_c,
2289 |             logistic_solver=logistic_solver,
2290 |             logistic_max_iter=args.logistic_max_iter,
2291 |             ablation_profile=args.ablation_profile,
2292 |             matched_shuffle_block_size=args.matched_shuffle_block_size,
2293 |         )
2294 |         all_rows.extend(bundle.rows)
2295 |         all_stats.append(bundle.group_loss_stats)
2296 | 
2297 |     unconverged = [
2298 |         row for row in all_rows if not row.converged
2299 |     ]
2300 |     if unconverged:
2301 |         names = sorted({row.feature_set for row in unconverged})
2302 |         print(
2303 |             "\nWARNING: Solver did not converge for: "
2304 |             + ", ".join(names)
2305 |             + ". Do not treat this run as final."
2306 |         )
2307 | 
2308 |     combined_stats = pd.concat(all_stats, ignore_index=True)
2309 |     summaries = build_summary_rows(
2310 |         all_rows,
2311 |         combined_stats,
2312 |         bootstrap_samples=args.bootstrap_samples,
2313 |         confidence_level=args.confidence_level,
2314 |         bootstrap_seed=args.bootstrap_seed,
2315 |     )
2316 | 
2317 |     print_results(all_rows, summaries)
2318 | 
2319 |     if args.out:
2320 |         output = pd.DataFrame(
2321 |             [dataclasses.asdict(row) for row in all_rows]
2322 |         )
2323 |         output.to_csv(args.out, index=False)
2324 |         print(f"\nSaved per-seed results to {args.out}")
2325 | 
2326 |     if args.summary_out:
2327 |         summary_output = pd.DataFrame(
2328 |             [dataclasses.asdict(row) for row in summaries]
2329 |         )
2330 |         summary_output.to_csv(args.summary_out, index=False)
2331 |         print(f"Saved summary results to {args.summary_out}")
2332 | 
2333 |     return 0
2334 | 
2335 | 
2336 | if __name__ == "__main__":
2337 |     raise SystemExit(main(sys.argv[1:]))
```


---

## F0003 — `probe.py`

```text
FILE_ID: F0003
PATH: probe.py
LANGUAGE: python
LINES: 1491
BYTES_UTF8: 48443
SHA256: c7bfe0cac947eb1e3e6baec1c7880822d181a38f77333bce521676d400dca896
```

```python
0001 | #!/usr/bin/env python3
0002 | """
0003 | PreferenceFutures probe, v2.
0004 | 
0005 | This script tests the claim:
0006 | 
0007 |     Preferences can contain incremental information about future outcomes.
0008 | 
0009 | For a future target F, history/candidate features H,A, and preference Y:
0010 | 
0011 |     PFI = Loss(F | H, A) - Loss(F | H, A, Y)
0012 | 
0013 | Positive PFI means that adding the observed preference improved held-out
0014 | forecasting under the evaluated model, dataset, split, and loss.
0015 | 
0016 | Version 2 adds the controls required for the synthetic appendix:
0017 | 
0018 | 1. Multiple random seeds.
0019 | 2. A configurable synthetic preference-to-future effect.
0020 | 3. A true null condition where preference has no future effect.
0021 | 4. Paired hierarchical bootstrap confidence intervals:
0022 |    seeds are resampled, then complete held-out sessions are resampled within
0023 |    each selected seed.
0024 | 5. Detailed and summary CSV outputs.
0025 | 
0026 | Tracks
0027 | ------
0028 | 
0029 | synthetic
0030 |     Controlled session-continuation data. Use
0031 |     --synthetic-preference-effects 0,0.25,0.5,0.75
0032 |     to verify that measured PFI is near zero under the null and increases as
0033 |     preference information is injected.
0034 | 
0035 | arena
0036 |     Loads lmarena-ai/arena-human-preference-140k from Hugging Face, groups rows
0037 |     by evaluation_session_id, orders them by evaluation_order, and predicts
0038 |     whether another evaluation follows the current vote.
0039 | 
0040 | Examples
0041 | --------
0042 | 
0043 | Synthetic null and positive controls across ten seeds:
0044 | 
0045 |     python preference_futures_probe.py \
0046 |       --track synthetic \
0047 |       --seeds 1,2,3,4,5,6,7,8,9,10 \
0048 |       --synthetic-preference-effects 0,0.75 \
0049 |       --bootstrap-samples 2000 \
0050 |       --out synthetic_runs.csv \
0051 |       --summary-out synthetic_summary.csv
0052 | 
0053 | Synthetic effect calibration curve:
0054 | 
0055 |     python preference_futures_probe.py \
0056 |       --track synthetic \
0057 |       --seeds 1,2,3,4,5,6,7,8,9,10 \
0058 |       --synthetic-preference-effects 0,0.25,0.5,0.75 \
0059 |       --bootstrap-samples 2000 \
0060 |       --out synthetic_effect_runs.csv \
0061 |       --summary-out synthetic_effect_summary.csv
0062 | 
0063 | Arena:
0064 | 
0065 |     HF_TOKEN=... python preference_futures_probe.py \
0066 |       --track arena \
0067 |       --seeds 1,2,3,4,5 \
0068 |       --bootstrap-samples 2000 \
0069 |       --out arena_runs.csv \
0070 |       --summary-out arena_summary.csv
0071 | 
0072 | Dependencies
0073 | ------------
0074 | 
0075 |     pip install pandas numpy scikit-learn datasets
0076 | """
0077 | 
0078 | from __future__ import annotations
0079 | 
0080 | import argparse
0081 | import dataclasses
0082 | import json
0083 | import math
0084 | import random
0085 | import re
0086 | import sys
0087 | from typing import Any, Iterable, Sequence
0088 | 
0089 | import numpy as np
0090 | import pandas as pd
0091 | 
0092 | 
0093 | # ---------------------------------------------------------------------------
0094 | # Data records
0095 | # ---------------------------------------------------------------------------
0096 | 
0097 | 
0098 | @dataclasses.dataclass
0099 | class ResultRow:
0100 |     track: str
0101 |     condition: str
0102 |     seed: int
0103 |     target: str
0104 |     feature_set: str
0105 |     n_train: int
0106 |     n_test: int
0107 |     n_train_groups: int
0108 |     n_test_groups: int
0109 |     loss_name: str
0110 |     loss: float
0111 |     brier: float | None = None
0112 |     auc: float | None = None
0113 |     accuracy: float | None = None
0114 |     synthetic_preference_effect: float | None = None
0115 |     synthetic_shared_latent_effect: float | None = None
0116 | 
0117 | 
0118 | @dataclasses.dataclass
0119 | class SummaryRow:
0120 |     track: str
0121 |     condition: str
0122 |     statistic: str
0123 |     n_seeds: int
0124 |     mean: float
0125 |     seed_std: float
0126 |     ci_low: float
0127 |     ci_high: float
0128 |     positive_seeds: int
0129 |     confidence_level: float
0130 |     bootstrap_samples: int
0131 |     synthetic_preference_effect: float | None = None
0132 |     synthetic_shared_latent_effect: float | None = None
0133 | 
0134 | 
0135 | @dataclasses.dataclass
0136 | class EvaluationBundle:
0137 |     rows: list[ResultRow]
0138 |     session_loss_stats: pd.DataFrame
0139 | 
0140 | 
0141 | # ---------------------------------------------------------------------------
0142 | # Utilities
0143 | # ---------------------------------------------------------------------------
0144 | 
0145 | 
0146 | def seed_everything(seed: int) -> None:
0147 |     random.seed(seed)
0148 |     np.random.seed(seed)
0149 | 
0150 | 
0151 | def sigmoid_scalar(value: float) -> float:
0152 |     if value >= 0:
0153 |         z = math.exp(-value)
0154 |         return 1.0 / (1.0 + z)
0155 |     z = math.exp(value)
0156 |     return z / (1.0 + z)
0157 | 
0158 | 
0159 | def parse_int_list(value: str | None, fallback: int) -> list[int]:
0160 |     if value is None or not value.strip():
0161 |         return [fallback]
0162 |     parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
0163 |     if not parsed:
0164 |         raise ValueError("--seeds did not contain any integers.")
0165 |     return parsed
0166 | 
0167 | 
0168 | def parse_float_list(value: str) -> list[float]:
0169 |     parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
0170 |     if not parsed:
0171 |         raise ValueError("The float list did not contain any values.")
0172 |     return parsed
0173 | 
0174 | 
0175 | def safe_jsonish_len(value: Any) -> int:
0176 |     if value is None:
0177 |         return 0
0178 |     if isinstance(value, float) and math.isnan(value):
0179 |         return 0
0180 |     if isinstance(value, str):
0181 |         return len(value)
0182 |     try:
0183 |         return len(json.dumps(value, ensure_ascii=False))
0184 |     except Exception:
0185 |         return len(str(value))
0186 | 
0187 | 
0188 | def safe_token_count(value: Any) -> int:
0189 |     if value is None:
0190 |         return 0
0191 |     if not isinstance(value, str):
0192 |         try:
0193 |             value = json.dumps(value, ensure_ascii=False)
0194 |         except Exception:
0195 |             value = str(value)
0196 |     return len(re.findall(r"\w+", value))
0197 | 
0198 | 
0199 | def normalise_winner(value: Any) -> str:
0200 |     if value is None:
0201 |         return "unknown"
0202 |     text = str(value).strip().lower()
0203 |     aliases = {
0204 |         "model_a": "a",
0205 |         "model a": "a",
0206 |         "winner_model_a": "a",
0207 |         "model_b": "b",
0208 |         "model b": "b",
0209 |         "winner_model_b": "b",
0210 |         "tie": "tie",
0211 |         "tie (bothbad)": "both_bad",
0212 |         "both_bad": "both_bad",
0213 |         "both bad": "both_bad",
0214 |     }
0215 |     if text in aliases:
0216 |         return aliases[text]
0217 |     if "both" in text and "bad" in text:
0218 |         return "both_bad"
0219 |     if text in {"a", "b"}:
0220 |         return text
0221 |     return text or "unknown"
0222 | 
0223 | 
0224 | def print_header(title: str) -> None:
0225 |     print("\n" + "=" * 96)
0226 |     print(title)
0227 |     print("=" * 96)
0228 | 
0229 | 
0230 | def log_loss_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
0231 |     p = np.clip(probabilities.astype(float), 1e-12, 1.0 - 1e-12)
0232 |     y = y_true.astype(float)
0233 |     return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
0234 | 
0235 | 
0236 | def brier_components(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
0237 |     return np.square(probabilities.astype(float) - y_true.astype(float))
0238 | 
0239 | 
0240 | # ---------------------------------------------------------------------------
0241 | # Model evaluation
0242 | # ---------------------------------------------------------------------------
0243 | 
0244 | 
0245 | def fit_binary_model(
0246 |     train_df: pd.DataFrame,
0247 |     test_df: pd.DataFrame,
0248 |     feature_columns: list[str],
0249 |     target_column: str,
0250 |     seed: int,
0251 | ) -> tuple[np.ndarray, dict[str, float]]:
0252 |     """Train a probability-forecasting logistic baseline.
0253 | 
0254 |     Important:
0255 |     ``class_weight="balanced"`` is deliberately NOT used here. Balanced class
0256 |     weights change the effective class prior seen by the optimiser. That can be
0257 |     useful when the objective is minority-class recall, but the raw
0258 |     ``predict_proba`` values no longer estimate probabilities under the
0259 |     observed data distribution. PFI is evaluated with proper probabilistic
0260 |     scoring rules (log loss and Brier score), so the model must be trained
0261 |     against the real class prevalence.
0262 | 
0263 |     The returned diagnostics include a constant-prevalence null model. A useful
0264 |     forecasting model should beat that null on log loss and Brier score.
0265 |     """
0266 |     from sklearn.compose import ColumnTransformer
0267 |     from sklearn.impute import SimpleImputer
0268 |     from sklearn.linear_model import LogisticRegression
0269 |     from sklearn.metrics import (
0270 |         accuracy_score,
0271 |         average_precision_score,
0272 |         balanced_accuracy_score,
0273 |         brier_score_loss,
0274 |         log_loss,
0275 |         roc_auc_score,
0276 |     )
0277 |     from sklearn.pipeline import Pipeline
0278 |     from sklearn.preprocessing import OneHotEncoder, StandardScaler
0279 | 
0280 |     X_train = train_df[feature_columns].copy()
0281 |     y_train = train_df[target_column].astype(int).to_numpy()
0282 |     X_test = test_df[feature_columns].copy()
0283 |     y_test = test_df[target_column].astype(int).to_numpy()
0284 | 
0285 |     if np.unique(y_train).size < 2:
0286 |         raise ValueError(
0287 |             f"Training split for target {target_column!r} contains only one class."
0288 |         )
0289 | 
0290 |     numeric_features = [
0291 |         column
0292 |         for column in feature_columns
0293 |         if pd.api.types.is_numeric_dtype(X_train[column])
0294 |     ]
0295 |     categorical_features = [
0296 |         column for column in feature_columns if column not in numeric_features
0297 |     ]
0298 | 
0299 |     numeric_pipe = Pipeline(
0300 |         steps=[
0301 |             ("imputer", SimpleImputer(strategy="median")),
0302 |             ("scaler", StandardScaler()),
0303 |         ]
0304 |     )
0305 |     categorical_pipe = Pipeline(
0306 |         steps=[
0307 |             ("imputer", SimpleImputer(strategy="most_frequent")),
0308 |             ("onehot", OneHotEncoder(handle_unknown="ignore")),
0309 |         ]
0310 |     )
0311 | 
0312 |     transformers: list[tuple[str, Any, list[str]]] = []
0313 |     if numeric_features:
0314 |         transformers.append(("num", numeric_pipe, numeric_features))
0315 |     if categorical_features:
0316 |         transformers.append(("cat", categorical_pipe, categorical_features))
0317 | 
0318 |     preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
0319 | 
0320 |     # Do not rebalance classes when the quantity under evaluation is a
0321 |     # probability under the observed distribution.
0322 |     model = LogisticRegression(
0323 |         max_iter=2000,
0324 |         class_weight=None,
0325 |         random_state=seed,
0326 |     )
0327 | 
0328 |     pipe = Pipeline(
0329 |         steps=[
0330 |             ("preprocessor", preprocessor),
0331 |             ("model", model),
0332 |         ]
0333 |     )
0334 | 
0335 |     pipe.fit(X_train, y_train)
0336 |     probabilities = pipe.predict_proba(X_test)[:, 1]
0337 |     predictions = (probabilities >= 0.5).astype(int)
0338 | 
0339 |     # Constant-probability baseline learned from the training split only.
0340 |     train_prevalence = float(np.mean(y_train))
0341 |     null_probabilities = np.full(
0342 |         shape=len(y_test),
0343 |         fill_value=np.clip(train_prevalence, 1e-12, 1.0 - 1e-12),
0344 |         dtype=float,
0345 |     )
0346 |     null_predictions = np.full(
0347 |         shape=len(y_test),
0348 |         fill_value=int(train_prevalence >= 0.5),
0349 |         dtype=int,
0350 |     )
0351 | 
0352 |     metrics: dict[str, float] = {
0353 |         "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
0354 |         "brier": float(brier_score_loss(y_test, probabilities)),
0355 |         "accuracy": float(accuracy_score(y_test, predictions)),
0356 |         "balanced_accuracy": float(
0357 |             balanced_accuracy_score(y_test, predictions)
0358 |         ),
0359 |         "average_precision": float(
0360 |             average_precision_score(y_test, probabilities)
0361 |         ),
0362 |         "train_prevalence": train_prevalence,
0363 |         "test_prevalence": float(np.mean(y_test)),
0364 |         "mean_predicted_probability": float(np.mean(probabilities)),
0365 |         "calibration_in_the_large": float(
0366 |             np.mean(probabilities) - np.mean(y_test)
0367 |         ),
0368 |         "null_log_loss": float(
0369 |             log_loss(y_test, null_probabilities, labels=[0, 1])
0370 |         ),
0371 |         "null_brier": float(
0372 |             brier_score_loss(y_test, null_probabilities)
0373 |         ),
0374 |         "null_accuracy": float(
0375 |             accuracy_score(y_test, null_predictions)
0376 |         ),
0377 |     }
0378 | 
0379 |     try:
0380 |         metrics["auc"] = float(roc_auc_score(y_test, probabilities))
0381 |     except ValueError:
0382 |         metrics["auc"] = float("nan")
0383 | 
0384 |     return probabilities, metrics
0385 | 
0386 | 
0387 | def shuffle_preference_columns(
0388 |     frame: pd.DataFrame,
0389 |     preference_features: list[str],
0390 |     seed: int,
0391 | ) -> pd.DataFrame:
0392 |     """Shuffle the one-hot preference vector within evaluation-order buckets."""
0393 |     shuffled = frame.copy()
0394 |     if "evaluation_order" in shuffled.columns:
0395 |         pieces: list[pd.DataFrame] = []
0396 |         for bucket_index, (_, part) in enumerate(
0397 |             shuffled.groupby("evaluation_order", dropna=False)
0398 |         ):
0399 |             values = part[preference_features].sample(
0400 |                 frac=1.0,
0401 |                 random_state=seed + bucket_index,
0402 |             ).to_numpy()
0403 |             pieces.append(
0404 |                 pd.DataFrame(
0405 |                     values,
0406 |                     columns=preference_features,
0407 |                     index=part.index,
0408 |                 )
0409 |             )
0410 |         shuffled_values = pd.concat(pieces).sort_index()
0411 |         shuffled.loc[:, preference_features] = shuffled_values
0412 |     else:
0413 |         rng = np.random.default_rng(seed)
0414 |         indices = rng.permutation(len(shuffled))
0415 |         shuffled.loc[:, preference_features] = (
0416 |             shuffled[preference_features].to_numpy()[indices]
0417 |         )
0418 |     return shuffled
0419 | 
0420 | 
0421 | def build_session_loss_stats(
0422 |     test_df: pd.DataFrame,
0423 |     *,
0424 |     group_column: str,
0425 |     target_column: str,
0426 |     seed: int,
0427 |     condition: str,
0428 |     predictions: dict[str, np.ndarray],
0429 | ) -> pd.DataFrame:
0430 |     """Build sufficient statistics for paired session-level bootstrapping."""
0431 |     y_true = test_df[target_column].astype(int).to_numpy()
0432 |     row_stats = pd.DataFrame(
0433 |         {
0434 |             "group_id": test_df[group_column].astype(str).to_numpy(),
0435 |             "n_rows": 1,
0436 |         },
0437 |         index=test_df.index,
0438 |     )
0439 | 
0440 |     for model_name, probability in predictions.items():
0441 |         row_stats[f"{model_name}_log_sum"] = log_loss_components(y_true, probability)
0442 |         row_stats[f"{model_name}_brier_sum"] = brier_components(y_true, probability)
0443 | 
0444 |     grouped = row_stats.groupby("group_id", as_index=False).sum(numeric_only=True)
0445 |     grouped["seed"] = seed
0446 |     grouped["condition"] = condition
0447 |     return grouped
0448 | 
0449 | 
0450 | def evaluate_binary_feature_sets(
0451 |     df: pd.DataFrame,
0452 |     *,
0453 |     group_column: str,
0454 |     target_column: str,
0455 |     base_features: list[str],
0456 |     preference_features: list[str],
0457 |     track: str,
0458 |     condition: str,
0459 |     seed: int,
0460 |     test_fraction: float = 0.2,
0461 |     synthetic_preference_effect: float | None = None,
0462 |     synthetic_shared_latent_effect: float | None = None,
0463 | ) -> EvaluationBundle:
0464 |     """Evaluate no-preference, preference-only, full, and shuffled controls."""
0465 |     rng = np.random.default_rng(seed)
0466 | 
0467 |     groups = np.asarray(sorted(df[group_column].dropna().astype(str).unique()))
0468 |     if groups.size < 2:
0469 |         raise ValueError("At least two groups are required for a grouped split.")
0470 |     rng.shuffle(groups)
0471 | 
0472 |     n_test_groups = max(1, int(round(len(groups) * test_fraction)))
0473 |     n_test_groups = min(n_test_groups, len(groups) - 1)
0474 |     test_groups = set(groups[:n_test_groups])
0475 | 
0476 |     group_values = df[group_column].astype(str)
0477 |     train_df = df[~group_values.isin(test_groups)].copy()
0478 |     test_df = df[group_values.isin(test_groups)].copy()
0479 | 
0480 |     if train_df.empty or test_df.empty:
0481 |         raise ValueError("Empty train/test split. Check grouping and dataset size.")
0482 | 
0483 |     feature_sets = {
0484 |         "history_candidate_no_preference": base_features,
0485 |         "preference_only": preference_features,
0486 |         "history_candidate_plus_preference": base_features + preference_features,
0487 |     }
0488 | 
0489 |     rows: list[ResultRow] = []
0490 |     predictions: dict[str, np.ndarray] = {}
0491 | 
0492 |     for name, columns in feature_sets.items():
0493 |         probability, metrics = fit_binary_model(
0494 |             train_df,
0495 |             test_df,
0496 |             columns,
0497 |             target_column,
0498 |             seed,
0499 |         )
0500 |         predictions[name] = probability
0501 |         rows.append(
0502 |             ResultRow(
0503 |                 track=track,
0504 |                 condition=condition,
0505 |                 seed=seed,
0506 |                 target=target_column,
0507 |                 feature_set=name,
0508 |                 n_train=len(train_df),
0509 |                 n_test=len(test_df),
0510 |                 n_train_groups=train_df[group_column].astype(str).nunique(),
0511 |                 n_test_groups=test_df[group_column].astype(str).nunique(),
0512 |                 loss_name="log_loss",
0513 |                 loss=metrics["log_loss"],
0514 |                 brier=metrics["brier"],
0515 |                 auc=metrics["auc"],
0516 |                 accuracy=metrics["accuracy"],
0517 |                 synthetic_preference_effect=synthetic_preference_effect,
0518 |                 synthetic_shared_latent_effect=synthetic_shared_latent_effect,
0519 |             )
0520 |         )
0521 | 
0522 |     shuffled_train = shuffle_preference_columns(
0523 |         train_df, preference_features, seed=seed + 10_000
0524 |     )
0525 |     shuffled_test = shuffle_preference_columns(
0526 |         test_df, preference_features, seed=seed + 20_000
0527 |     )
0528 | 
0529 |     shuffled_probability, shuffled_metrics = fit_binary_model(
0530 |         shuffled_train,
0531 |         shuffled_test,
0532 |         base_features + preference_features,
0533 |         target_column,
0534 |         seed,
0535 |     )
0536 |     shuffled_name = "history_candidate_plus_shuffled_preference"
0537 |     predictions[shuffled_name] = shuffled_probability
0538 |     rows.append(
0539 |         ResultRow(
0540 |             track=track,
0541 |             condition=condition,
0542 |             seed=seed,
0543 |             target=target_column,
0544 |             feature_set=shuffled_name,
0545 |             n_train=len(shuffled_train),
0546 |             n_test=len(shuffled_test),
0547 |             n_train_groups=shuffled_train[group_column].astype(str).nunique(),
0548 |             n_test_groups=shuffled_test[group_column].astype(str).nunique(),
0549 |             loss_name="log_loss",
0550 |             loss=shuffled_metrics["log_loss"],
0551 |             brier=shuffled_metrics["brier"],
0552 |             auc=shuffled_metrics["auc"],
0553 |             accuracy=shuffled_metrics["accuracy"],
0554 |             synthetic_preference_effect=synthetic_preference_effect,
0555 |             synthetic_shared_latent_effect=synthetic_shared_latent_effect,
0556 |         )
0557 |     )
0558 | 
0559 |     stats = build_session_loss_stats(
0560 |         test_df,
0561 |         group_column=group_column,
0562 |         target_column=target_column,
0563 |         seed=seed,
0564 |         condition=condition,
0565 |         predictions={
0566 |             "no_pref": predictions["history_candidate_no_preference"],
0567 |             "full": predictions["history_candidate_plus_preference"],
0568 |             "shuffled": predictions[shuffled_name],
0569 |         },
0570 |     )
0571 | 
0572 |     return EvaluationBundle(rows=rows, session_loss_stats=stats)
0573 | 
0574 | 
0575 | # ---------------------------------------------------------------------------
0576 | # Bootstrap and aggregation
0577 | # ---------------------------------------------------------------------------
0578 | 
0579 | 
0580 | def statistic_from_session_stats(stats: pd.DataFrame, statistic: str) -> float:
0581 |     total_rows = float(stats["n_rows"].sum())
0582 |     if total_rows <= 0:
0583 |         return float("nan")
0584 | 
0585 |     if statistic == "pfi_log_loss":
0586 |         return float(
0587 |             (
0588 |                 stats["no_pref_log_sum"].sum()
0589 |                 - stats["full_log_sum"].sum()
0590 |             )
0591 |             / total_rows
0592 |         )
0593 |     if statistic == "pfi_brier":
0594 |         return float(
0595 |             (
0596 |                 stats["no_pref_brier_sum"].sum()
0597 |                 - stats["full_brier_sum"].sum()
0598 |             )
0599 |             / total_rows
0600 |         )
0601 |     if statistic == "shuffle_gap_log_loss":
0602 |         return float(
0603 |             (
0604 |                 stats["shuffled_log_sum"].sum()
0605 |                 - stats["full_log_sum"].sum()
0606 |             )
0607 |             / total_rows
0608 |         )
0609 |     if statistic == "shuffle_gap_brier":
0610 |         return float(
0611 |             (
0612 |                 stats["shuffled_brier_sum"].sum()
0613 |                 - stats["full_brier_sum"].sum()
0614 |             )
0615 |             / total_rows
0616 |         )
0617 |     raise ValueError(f"Unknown statistic: {statistic}")
0618 | 
0619 | 
0620 | def hierarchical_bootstrap_interval(
0621 |     stats: pd.DataFrame,
0622 |     *,
0623 |     statistic: str,
0624 |     samples: int,
0625 |     confidence_level: float,
0626 |     seed: int,
0627 | ) -> tuple[float, float]:
0628 |     """Resample seeds, then complete held-out sessions within each seed.
0629 | 
0630 |     This implementation operates on NumPy sufficient-statistic arrays rather
0631 |     than repeatedly concatenating pandas frames, keeping publication-sized
0632 |     bootstrap runs practical.
0633 |     """
0634 |     if samples <= 0:
0635 |         return float("nan"), float("nan")
0636 |     if not 0.0 < confidence_level < 1.0:
0637 |         raise ValueError("--confidence-level must be between 0 and 1.")
0638 | 
0639 |     rng = np.random.default_rng(seed)
0640 |     seed_values = np.asarray(sorted(stats["seed"].unique()), dtype=int)
0641 |     if seed_values.size == 0:
0642 |         return float("nan"), float("nan")
0643 | 
0644 |     columns = [
0645 |         "n_rows",
0646 |         "no_pref_log_sum",
0647 |         "full_log_sum",
0648 |         "shuffled_log_sum",
0649 |         "no_pref_brier_sum",
0650 |         "full_brier_sum",
0651 |         "shuffled_brier_sum",
0652 |     ]
0653 |     by_seed = {
0654 |         int(seed_value): stats.loc[
0655 |             stats["seed"] == seed_value, columns
0656 |         ].to_numpy(dtype=float)
0657 |         for seed_value in seed_values
0658 |     }
0659 | 
0660 |     def from_totals(totals: np.ndarray) -> float:
0661 |         n_rows = totals[0]
0662 |         if n_rows <= 0:
0663 |             return float("nan")
0664 |         if statistic == "pfi_log_loss":
0665 |             return float((totals[1] - totals[2]) / n_rows)
0666 |         if statistic == "shuffle_gap_log_loss":
0667 |             return float((totals[3] - totals[2]) / n_rows)
0668 |         if statistic == "pfi_brier":
0669 |             return float((totals[4] - totals[5]) / n_rows)
0670 |         if statistic == "shuffle_gap_brier":
0671 |             return float((totals[6] - totals[5]) / n_rows)
0672 |         raise ValueError(f"Unknown statistic: {statistic}")
0673 | 
0674 |     draws = np.empty(samples, dtype=float)
0675 |     for draw_index in range(samples):
0676 |         sampled_seed_values = rng.choice(
0677 |             seed_values, size=len(seed_values), replace=True
0678 |         )
0679 |         totals = np.zeros(len(columns), dtype=float)
0680 |         for sampled_seed in sampled_seed_values:
0681 |             seed_array = by_seed[int(sampled_seed)]
0682 |             row_indices = rng.integers(0, len(seed_array), size=len(seed_array))
0683 |             totals += seed_array[row_indices].sum(axis=0)
0684 |         draws[draw_index] = from_totals(totals)
0685 | 
0686 |     alpha = 1.0 - confidence_level
0687 |     low = float(np.quantile(draws, alpha / 2.0))
0688 |     high = float(np.quantile(draws, 1.0 - alpha / 2.0))
0689 |     return low, high
0690 | 
0691 | 
0692 | def build_summary_rows(
0693 |     detailed_rows: list[ResultRow],
0694 |     session_stats: pd.DataFrame,
0695 |     *,
0696 |     track: str,
0697 |     condition: str,
0698 |     bootstrap_samples: int,
0699 |     confidence_level: float,
0700 |     bootstrap_seed: int,
0701 |     synthetic_preference_effect: float | None,
0702 |     synthetic_shared_latent_effect: float | None,
0703 | ) -> list[SummaryRow]:
0704 |     result_df = pd.DataFrame([dataclasses.asdict(row) for row in detailed_rows])
0705 |     seed_values = sorted(result_df["seed"].unique())
0706 | 
0707 |     per_seed: dict[str, list[float]] = {
0708 |         "pfi_log_loss": [],
0709 |         "pfi_brier": [],
0710 |         "shuffle_gap_log_loss": [],
0711 |         "shuffle_gap_brier": [],
0712 |     }
0713 | 
0714 |     for seed_value in seed_values:
0715 |         subset = result_df[result_df["seed"] == seed_value]
0716 |         indexed = subset.set_index("feature_set")
0717 |         no_pref = indexed.loc["history_candidate_no_preference"]
0718 |         full = indexed.loc["history_candidate_plus_preference"]
0719 |         shuffled = indexed.loc["history_candidate_plus_shuffled_preference"]
0720 | 
0721 |         per_seed["pfi_log_loss"].append(float(no_pref["loss"] - full["loss"]))
0722 |         per_seed["pfi_brier"].append(float(no_pref["brier"] - full["brier"]))
0723 |         per_seed["shuffle_gap_log_loss"].append(
0724 |             float(shuffled["loss"] - full["loss"])
0725 |         )
0726 |         per_seed["shuffle_gap_brier"].append(
0727 |             float(shuffled["brier"] - full["brier"])
0728 |         )
0729 | 
0730 |     rows: list[SummaryRow] = []
0731 |     for statistic, values_list in per_seed.items():
0732 |         values = np.asarray(values_list, dtype=float)
0733 |         low, high = hierarchical_bootstrap_interval(
0734 |             session_stats,
0735 |             statistic=statistic,
0736 |             samples=bootstrap_samples,
0737 |             confidence_level=confidence_level,
0738 |             seed=bootstrap_seed + sum(ord(ch) for ch in statistic),
0739 |         )
0740 |         rows.append(
0741 |             SummaryRow(
0742 |                 track=track,
0743 |                 condition=condition,
0744 |                 statistic=statistic,
0745 |                 n_seeds=len(values),
0746 |                 mean=float(np.mean(values)),
0747 |                 seed_std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
0748 |                 ci_low=low,
0749 |                 ci_high=high,
0750 |                 positive_seeds=int(np.sum(values > 0)),
0751 |                 confidence_level=confidence_level,
0752 |                 bootstrap_samples=bootstrap_samples,
0753 |                 synthetic_preference_effect=synthetic_preference_effect,
0754 |                 synthetic_shared_latent_effect=synthetic_shared_latent_effect,
0755 |             )
0756 |         )
0757 |     return rows
0758 | 
0759 | 
0760 | def print_condition_results(
0761 |     rows: list[ResultRow],
0762 |     summary_rows: list[SummaryRow],
0763 | ) -> None:
0764 |     result_df = pd.DataFrame([dataclasses.asdict(row) for row in rows])
0765 |     metric_columns = ["loss", "brier", "auc", "accuracy"]
0766 |     aggregate = (
0767 |         result_df.groupby("feature_set")[metric_columns]
0768 |         .agg(["mean", "std"])
0769 |         .sort_index()
0770 |     )
0771 | 
0772 |     print("\nFeature-set metrics across seeds")
0773 |     print(aggregate.to_string(float_format=lambda value: f"{value:.6f}"))
0774 | 
0775 |     summary_df = pd.DataFrame(
0776 |         [dataclasses.asdict(row) for row in summary_rows]
0777 |     )
0778 |     display_columns = [
0779 |         "statistic",
0780 |         "n_seeds",
0781 |         "mean",
0782 |         "seed_std",
0783 |         "ci_low",
0784 |         "ci_high",
0785 |         "positive_seeds",
0786 |     ]
0787 |     print("\nPFI and shuffled-control summary")
0788 |     print(
0789 |         summary_df[display_columns].to_string(
0790 |             index=False,
0791 |             float_format=lambda value: f"{value:.6f}",
0792 |         )
0793 |     )
0794 | 
0795 | 
0796 | # ---------------------------------------------------------------------------
0797 | # Synthetic track
0798 | # ---------------------------------------------------------------------------
0799 | 
0800 | 
0801 | def make_synthetic_dataset(
0802 |     n_sessions: int,
0803 |     rows_per_session: int,
0804 |     seed: int,
0805 |     preference_effect: float,
0806 |     shared_latent_effect: float,
0807 | ) -> pd.DataFrame:
0808 |     """Generate controlled future-bearing preference trajectories.
0809 | 
0810 |     Preference generation:
0811 |         Y depends on visible candidate features and a latent session regime.
0812 | 
0813 |     Future generation:
0814 |         F depends on visible context, independent patience, and
0815 |         preference_effect * Y.
0816 | 
0817 |     By default shared_latent_effect is zero. Therefore, when
0818 |     preference_effect is also zero, Y is conditionally independent of F given
0819 |     the visible base features: this is the strict null condition.
0820 | 
0821 |     Setting shared_latent_effect above zero creates an optional observational
0822 |     confounding experiment in which the preference reveals additional
0823 |     information about a latent regime that also affects the future.
0824 |     """
0825 |     rng = np.random.default_rng(seed)
0826 |     records: list[dict[str, Any]] = []
0827 | 
0828 |     for session_id in range(n_sessions):
0829 |         latent_regime = float(rng.normal())
0830 |         patience = float(rng.normal())
0831 |         session_topic = str(
0832 |             rng.choice(["code", "writing", "reasoning", "qa"])
0833 |         )
0834 | 
0835 |         for order in range(rows_per_session):
0836 |             difficulty = float(rng.normal() + 0.25 * order)
0837 |             response_gap = float(rng.normal() + 0.45 * latent_regime)
0838 |             candidate_length_delta = float(rng.normal() + 0.2 * difficulty)
0839 |             history_length = max(
0840 |                 1, int(100 + 20 * order + rng.normal(0, 15))
0841 |             )
0842 | 
0843 |             preference_logit = (
0844 |                 0.8 * response_gap
0845 |                 - 0.25 * candidate_length_delta
0846 |                 + 0.65 * latent_regime
0847 |                 + float(rng.normal(0, 0.75))
0848 |             )
0849 |             preference_a = int(
0850 |                 rng.random() < sigmoid_scalar(preference_logit)
0851 |             )
0852 | 
0853 |             future_logit = (
0854 |                 -0.25 * order
0855 |                 + 0.45 * difficulty
0856 |                 + 0.35 * patience
0857 |                 + preference_effect * preference_a
0858 |                 + shared_latent_effect * latent_regime
0859 |                 + float(rng.normal(0, 0.8))
0860 |             )
0861 |             continues = int(rng.random() < sigmoid_scalar(future_logit))
0862 | 
0863 |             records.append(
0864 |                 {
0865 |                     "evaluation_session_id": f"s{session_id}",
0866 |                     "evaluation_order": order,
0867 |                     "topic": session_topic,
0868 |                     "history_len": history_length,
0869 |                     "candidate_a_len": (
0870 |                         120 + 20 * response_gap + rng.normal(0, 10)
0871 |                     ),
0872 |                     "candidate_b_len": (
0873 |                         120 - 20 * response_gap + rng.normal(0, 10)
0874 |                     ),
0875 |                     "candidate_len_delta": candidate_length_delta,
0876 |                     "response_gap_proxy": response_gap,
0877 |                     "difficulty_proxy": difficulty,
0878 |                     "winner_norm": "a" if preference_a else "b",
0879 |                     "pref_a": preference_a,
0880 |                     "pref_b": 1 - preference_a,
0881 |                     "pref_tie": 0,
0882 |                     "pref_both_bad": 0,
0883 |                     "session_continues_after_vote": continues,
0884 |                 }
0885 |             )
0886 | 
0887 |             if not continues:
0888 |                 break
0889 | 
0890 |     return pd.DataFrame(records)
0891 | 
0892 | 
0893 | def synthetic_features() -> tuple[list[str], list[str]]:
0894 |     base_features = [
0895 |         "evaluation_order",
0896 |         "topic",
0897 |         "history_len",
0898 |         "candidate_a_len",
0899 |         "candidate_b_len",
0900 |         "candidate_len_delta",
0901 |         "response_gap_proxy",
0902 |         "difficulty_proxy",
0903 |     ]
0904 |     preference_features = ["pref_a", "pref_b", "pref_tie", "pref_both_bad"]
0905 |     return base_features, preference_features
0906 | 
0907 | 
0908 | def run_synthetic_matrix(
0909 |     args: argparse.Namespace,
0910 |     seeds: Sequence[int],
0911 | ) -> tuple[list[ResultRow], list[SummaryRow]]:
0912 |     effects = parse_float_list(args.synthetic_preference_effects)
0913 |     base_features, preference_features = synthetic_features()
0914 | 
0915 |     all_rows: list[ResultRow] = []
0916 |     all_summaries: list[SummaryRow] = []
0917 | 
0918 |     for effect_index, effect in enumerate(effects):
0919 |         condition = (
0920 |             f"synthetic_effect_{effect:g}"
0921 |             f"_shared_latent_{args.synthetic_shared_latent_effect:g}"
0922 |         )
0923 |         condition_rows: list[ResultRow] = []
0924 |         condition_stats: list[pd.DataFrame] = []
0925 | 
0926 |         print_header(
0927 |             f"Synthetic condition: preference effect={effect:g}, "
0928 |             f"shared latent effect={args.synthetic_shared_latent_effect:g}"
0929 |         )
0930 | 
0931 |         for seed in seeds:
0932 |             df = make_synthetic_dataset(
0933 |                 n_sessions=args.synthetic_sessions,
0934 |                 rows_per_session=args.synthetic_max_rounds,
0935 |                 seed=seed,
0936 |                 preference_effect=effect,
0937 |                 shared_latent_effect=args.synthetic_shared_latent_effect,
0938 |             )
0939 |             bundle = evaluate_binary_feature_sets(
0940 |                 df,
0941 |                 group_column="evaluation_session_id",
0942 |                 target_column="session_continues_after_vote",
0943 |                 base_features=base_features,
0944 |                 preference_features=preference_features,
0945 |                 track="synthetic",
0946 |                 condition=condition,
0947 |                 seed=seed,
0948 |                 test_fraction=args.test_fraction,
0949 |                 synthetic_preference_effect=effect,
0950 |                 synthetic_shared_latent_effect=args.synthetic_shared_latent_effect,
0951 |             )
0952 |             condition_rows.extend(bundle.rows)
0953 |             condition_stats.append(bundle.session_loss_stats)
0954 | 
0955 |             target_rate = float(
0956 |                 df["session_continues_after_vote"].mean()
0957 |             )
0958 |             print(
0959 |                 f"seed={seed:>4} rows={len(df):>6,} "
0960 |                 f"sessions={df['evaluation_session_id'].nunique():>5,} "
0961 |                 f"continuation_rate={target_rate:.4f}"
0962 |             )
0963 | 
0964 |         combined_stats = pd.concat(condition_stats, ignore_index=True)
0965 |         summaries = build_summary_rows(
0966 |             condition_rows,
0967 |             combined_stats,
0968 |             track="synthetic",
0969 |             condition=condition,
0970 |             bootstrap_samples=args.bootstrap_samples,
0971 |             confidence_level=args.confidence_level,
0972 |             bootstrap_seed=args.bootstrap_seed + effect_index * 1000,
0973 |             synthetic_preference_effect=effect,
0974 |             synthetic_shared_latent_effect=args.synthetic_shared_latent_effect,
0975 |         )
0976 | 
0977 |         print_condition_results(condition_rows, summaries)
0978 |         all_rows.extend(condition_rows)
0979 |         all_summaries.extend(summaries)
0980 | 
0981 |     return all_rows, all_summaries
0982 | 
0983 | 
0984 | # ---------------------------------------------------------------------------
0985 | # Arena track
0986 | # ---------------------------------------------------------------------------
0987 | 
0988 | 
0989 | def load_arena_dataframe(
0990 |     limit_rows: int | None,
0991 |     *,
0992 |     sample_seed: int = 1729,
0993 | ) -> pd.DataFrame:
0994 |     """Load Arena data, preserving complete evaluation sessions.
0995 | 
0996 |     The old smoke-test implementation selected the first ``limit_rows`` rows
0997 |     before constructing the continuation target. That can cut a session in
0998 |     half and falsely label its last retained row as a terminal event.
0999 | 
1000 |     This version loads the session-id column first, samples complete sessions,
1001 |     and only then materialises the selected rows.
1002 |     """
1003 |     try:
1004 |         from datasets import load_dataset
1005 |     except ImportError as exc:
1006 |         raise SystemExit(
1007 |             "Missing dependency: datasets. Install with `pip install datasets`."
1008 |         ) from exc
1009 | 
1010 |     repo_id = "lmarena-ai/arena-human-preference-140k"
1011 |     dataset = load_dataset(repo_id, split="train", token=True)
1012 | 
1013 |     if limit_rows is not None and limit_rows > 0 and limit_rows < len(dataset):
1014 |         session_ids = np.asarray(
1015 |             dataset["evaluation_session_id"],
1016 |             dtype=object,
1017 |         )
1018 | 
1019 |         unique_sessions, counts = np.unique(
1020 |             session_ids,
1021 |             return_counts=True,
1022 |         )
1023 |         rng = np.random.default_rng(sample_seed)
1024 |         order = rng.permutation(len(unique_sessions))
1025 | 
1026 |         selected_sessions: list[object] = []
1027 |         selected_rows = 0
1028 |         for index in order:
1029 |             session_size = int(counts[index])
1030 |             if selected_sessions and selected_rows + session_size > limit_rows:
1031 |                 continue
1032 |             selected_sessions.append(unique_sessions[index])
1033 |             selected_rows += session_size
1034 |             if selected_rows >= limit_rows:
1035 |                 break
1036 | 
1037 |         selected_set = set(selected_sessions)
1038 |         selected_indices = [
1039 |             index
1040 |             for index, session_id in enumerate(session_ids)
1041 |             if session_id in selected_set
1042 |         ]
1043 |         dataset = dataset.select(selected_indices)
1044 | 
1045 |     return dataset.to_pandas()
1046 | 
1047 | 
1048 | def pick_first_existing(
1049 |     columns: Iterable[str], candidates: list[str]
1050 | ) -> str | None:
1051 |     available = set(columns)
1052 |     for candidate in candidates:
1053 |         if candidate in available:
1054 |             return candidate
1055 |     return None
1056 | 
1057 | 
1058 | def prepare_arena_future_dataset(
1059 |     raw: pd.DataFrame,
1060 | ) -> tuple[pd.DataFrame, list[str], list[str]]:
1061 |     required_candidates = {
1062 |         "evaluation_session_id": ["evaluation_session_id", "session_id"],
1063 |         "evaluation_order": ["evaluation_order", "order", "turn"],
1064 |         "winner": ["winner", "vote", "preference"],
1065 |     }
1066 | 
1067 |     resolved: dict[str, str] = {}
1068 |     for logical_name, candidate_names in required_candidates.items():
1069 |         found = pick_first_existing(raw.columns, candidate_names)
1070 |         if found is None:
1071 |             schema = "\n".join(f"  - {column}" for column in raw.columns)
1072 |             raise ValueError(
1073 |                 f"Could not find required logical field {logical_name!r}. "
1074 |                 f"Available columns:\n{schema}"
1075 |             )
1076 |         resolved[logical_name] = found
1077 | 
1078 |     df = raw.copy()
1079 |     df["_session_id"] = df[resolved["evaluation_session_id"]].astype(str)
1080 |     df["_order"] = pd.to_numeric(
1081 |         df[resolved["evaluation_order"]], errors="coerce"
1082 |     )
1083 |     df = df.dropna(subset=["_order"]).copy()
1084 |     df["_order"] = df["_order"].astype(int)
1085 |     df["winner_norm"] = df[resolved["winner"]].map(normalise_winner)
1086 | 
1087 |     df = df.sort_values(["_session_id", "_order"]).copy()
1088 |     group_sizes = df.groupby("_session_id")["_order"].transform("size")
1089 |     group_rank = df.groupby("_session_id").cumcount()
1090 |     df["session_continues_after_vote"] = (
1091 |         group_rank < group_sizes - 1
1092 |     ).astype(int)
1093 | 
1094 |     conversation_a = pick_first_existing(
1095 |         df.columns,
1096 |         ["conversation_a", "messages_a", "response_a", "answer_a"],
1097 |     )
1098 |     conversation_b = pick_first_existing(
1099 |         df.columns,
1100 |         ["conversation_b", "messages_b", "response_b", "answer_b"],
1101 |     )
1102 |     full_conversation = pick_first_existing(
1103 |         df.columns,
1104 |         ["full_conversation", "conversation", "messages", "prompt"],
1105 |     )
1106 |     model_a = pick_first_existing(
1107 |         df.columns, ["model_a", "model_a_name"]
1108 |     )
1109 |     model_b = pick_first_existing(
1110 |         df.columns, ["model_b", "model_b_name"]
1111 |     )
1112 |     category = pick_first_existing(
1113 |         df.columns,
1114 |         ["category", "categories", "turn_category", "language"],
1115 |     )
1116 | 
1117 |     df["candidate_a_chars"] = (
1118 |         df[conversation_a].map(safe_jsonish_len) if conversation_a else 0
1119 |     )
1120 |     df["candidate_a_tokens"] = (
1121 |         df[conversation_a].map(safe_token_count) if conversation_a else 0
1122 |     )
1123 |     df["candidate_b_chars"] = (
1124 |         df[conversation_b].map(safe_jsonish_len) if conversation_b else 0
1125 |     )
1126 |     df["candidate_b_tokens"] = (
1127 |         df[conversation_b].map(safe_token_count) if conversation_b else 0
1128 |     )
1129 | 
1130 |     if full_conversation:
1131 |         df["history_chars"] = df[full_conversation].map(safe_jsonish_len)
1132 |         df["history_tokens"] = df[full_conversation].map(safe_token_count)
1133 |     else:
1134 |         df["history_chars"] = (
1135 |             df["candidate_a_chars"] + df["candidate_b_chars"]
1136 |         )
1137 |         df["history_tokens"] = (
1138 |             df["candidate_a_tokens"] + df["candidate_b_tokens"]
1139 |         )
1140 | 
1141 |     df["candidate_char_delta"] = (
1142 |         df["candidate_a_chars"] - df["candidate_b_chars"]
1143 |     )
1144 |     df["candidate_token_delta"] = (
1145 |         df["candidate_a_tokens"] - df["candidate_b_tokens"]
1146 |     )
1147 | 
1148 |     df["model_a_feature"] = (
1149 |         df[model_a].astype(str) if model_a else "unknown"
1150 |     )
1151 |     df["model_b_feature"] = (
1152 |         df[model_b].astype(str) if model_b else "unknown"
1153 |     )
1154 |     df["category_feature"] = (
1155 |         df[category].astype(str) if category else "unknown"
1156 |     )
1157 | 
1158 |     for label in ["a", "b", "tie", "both_bad"]:
1159 |         df[f"pref_{label}"] = (
1160 |             df["winner_norm"] == label
1161 |         ).astype(int)
1162 | 
1163 |     # Canonicalise without renaming temporary columns onto existing source
1164 |     # columns. The Arena dataset already uses ``evaluation_session_id`` and
1165 |     # ``evaluation_order``; renaming ``_session_id``/``_order`` to those names
1166 |     # would create duplicate column labels. In pandas, selecting a duplicated
1167 |     # label returns a DataFrame rather than a Series, which breaks operations
1168 |     # such as ``value_counts()`` and ``groupby()``.
1169 |     prepared = df.copy()
1170 |     prepared["evaluation_session_id"] = prepared["_session_id"].astype(str)
1171 |     prepared["evaluation_order"] = prepared["_order"].astype(int)
1172 |     prepared = prepared.drop(columns=["_session_id", "_order"])
1173 | 
1174 |     duplicate_columns = prepared.columns[prepared.columns.duplicated()].tolist()
1175 |     if duplicate_columns:
1176 |         raise ValueError(
1177 |             "Arena preparation produced duplicate columns: "
1178 |             f"{duplicate_columns}. Available columns: {list(prepared.columns)}"
1179 |         )
1180 | 
1181 |     base_features = [
1182 |         "evaluation_order",
1183 |         "history_chars",
1184 |         "history_tokens",
1185 |         "candidate_a_chars",
1186 |         "candidate_b_chars",
1187 |         "candidate_a_tokens",
1188 |         "candidate_b_tokens",
1189 |         "candidate_char_delta",
1190 |         "candidate_token_delta",
1191 |         "model_a_feature",
1192 |         "model_b_feature",
1193 |         "category_feature",
1194 |     ]
1195 |     preference_features = ["pref_a", "pref_b", "pref_tie", "pref_both_bad"]
1196 |     return prepared, base_features, preference_features
1197 | 
1198 | 
1199 | 
1200 | def print_arena_target_diagnostics(df: pd.DataFrame) -> None:
1201 |     """Print direct evidence about whether the vote predicts continuation.
1202 | 
1203 |     These diagnostics are intentionally model-free. They reveal whether the raw
1204 |     vote categories have any visible association with the future target before
1205 |     a classifier is fitted.
1206 |     """
1207 |     from sklearn.metrics import brier_score_loss, log_loss
1208 | 
1209 |     target = "session_continues_after_vote"
1210 |     y = df[target].astype(int).to_numpy()
1211 |     prevalence = float(np.mean(y))
1212 |     constant_probability = np.full(
1213 |         len(y),
1214 |         np.clip(prevalence, 1e-12, 1.0 - 1e-12),
1215 |         dtype=float,
1216 |     )
1217 | 
1218 |     print_header("Arena target diagnostics")
1219 |     print(f"Rows: {len(df):,}")
1220 |     print(f"Positive continuation rate: {prevalence:.6f}")
1221 |     print(f"Always-stop accuracy: {1.0 - prevalence:.6f}")
1222 |     print(
1223 |         "Constant-prevalence log loss: "
1224 |         f"{log_loss(y, constant_probability, labels=[0, 1]):.6f}"
1225 |     )
1226 |     print(
1227 |         "Constant-prevalence Brier score: "
1228 |         f"{brier_score_loss(y, constant_probability):.6f}"
1229 |     )
1230 | 
1231 |     vote_table = (
1232 |         df.groupby("winner_norm", dropna=False)[target]
1233 |         .agg(["count", "sum", "mean"])
1234 |         .rename(
1235 |             columns={
1236 |                 "sum": "continuations",
1237 |                 "mean": "continuation_rate",
1238 |             }
1239 |         )
1240 |         .sort_values("continuation_rate", ascending=False)
1241 |     )
1242 |     vote_table["lift_vs_overall"] = (
1243 |         vote_table["continuation_rate"] - prevalence
1244 |     )
1245 |     print("\nContinuation by current vote")
1246 |     print(vote_table.to_string(float_format=lambda value: f"{value:.6f}"))
1247 | 
1248 |     minimum_order = int(df["evaluation_order"].min())
1249 |     first = df[df["evaluation_order"] == minimum_order]
1250 |     if not first.empty:
1251 |         first_prevalence = float(first[target].mean())
1252 |         first_table = (
1253 |             first.groupby("winner_norm", dropna=False)[target]
1254 |             .agg(["count", "sum", "mean"])
1255 |             .rename(
1256 |                 columns={
1257 |                     "sum": "continuations",
1258 |                     "mean": "continuation_rate",
1259 |                 }
1260 |             )
1261 |             .sort_values("continuation_rate", ascending=False)
1262 |         )
1263 |         first_table["lift_vs_first_round"] = (
1264 |             first_table["continuation_rate"] - first_prevalence
1265 |         )
1266 |         print(f"\nContinuation by vote at evaluation_order={minimum_order}")
1267 |         print(
1268 |             first_table.to_string(
1269 |                 float_format=lambda value: f"{value:.6f}"
1270 |             )
1271 |         )
1272 | 
1273 |     order_table = (
1274 |         df.groupby("evaluation_order")[target]
1275 |         .agg(["count", "mean"])
1276 |         .rename(columns={"mean": "continuation_rate"})
1277 |         .head(12)
1278 |     )
1279 |     print("\nContinuation by evaluation order")
1280 |     print(order_table.to_string(float_format=lambda value: f"{value:.6f}"))
1281 | 
1282 |     session_sizes = df.groupby("evaluation_session_id").size()
1283 |     print("\nSession completeness summary")
1284 |     print(session_sizes.describe().to_string())
1285 | 
1286 | 
1287 | def run_arena_matrix(
1288 |     args: argparse.Namespace,
1289 |     seeds: Sequence[int],
1290 | ) -> tuple[list[ResultRow], list[SummaryRow]]:
1291 |     raw = load_arena_dataframe(
1292 |         args.limit_rows,
1293 |         sample_seed=args.bootstrap_seed,
1294 |     )
1295 |     df, base_features, preference_features = prepare_arena_future_dataset(raw)
1296 |     print_arena_target_diagnostics(df)
1297 | 
1298 |     print_header("Arena dataset audit")
1299 |     session_counts = df["evaluation_session_id"].value_counts()
1300 |     print(f"Rows: {len(df):,}")
1301 |     print(f"Sessions: {df['evaluation_session_id'].nunique():,}")
1302 |     print(session_counts.describe().to_string())
1303 |     print("Sessions with 2+ evaluations:", int((session_counts >= 2).sum()))
1304 |     print("Sessions with 3+ evaluations:", int((session_counts >= 3).sum()))
1305 |     print("Sessions with 5+ evaluations:", int((session_counts >= 5).sum()))
1306 |     print("\nWinner distribution:")
1307 |     print(df["winner_norm"].value_counts(dropna=False).to_string())
1308 |     print("\nContinuation target distribution:")
1309 |     print(
1310 |         df["session_continues_after_vote"]
1311 |         .value_counts(normalize=True)
1312 |         .to_string()
1313 |     )
1314 | 
1315 |     condition = "arena"
1316 |     condition_rows: list[ResultRow] = []
1317 |     condition_stats: list[pd.DataFrame] = []
1318 | 
1319 |     for seed in seeds:
1320 |         bundle = evaluate_binary_feature_sets(
1321 |             df,
1322 |             group_column="evaluation_session_id",
1323 |             target_column="session_continues_after_vote",
1324 |             base_features=base_features,
1325 |             preference_features=preference_features,
1326 |             track="arena",
1327 |             condition=condition,
1328 |             seed=seed,
1329 |             test_fraction=args.test_fraction,
1330 |         )
1331 |         condition_rows.extend(bundle.rows)
1332 |         condition_stats.append(bundle.session_loss_stats)
1333 | 
1334 |     combined_stats = pd.concat(condition_stats, ignore_index=True)
1335 |     summaries = build_summary_rows(
1336 |         condition_rows,
1337 |         combined_stats,
1338 |         track="arena",
1339 |         condition=condition,
1340 |         bootstrap_samples=args.bootstrap_samples,
1341 |         confidence_level=args.confidence_level,
1342 |         bootstrap_seed=args.bootstrap_seed,
1343 |         synthetic_preference_effect=None,
1344 |         synthetic_shared_latent_effect=None,
1345 |     )
1346 |     print_condition_results(condition_rows, summaries)
1347 |     return condition_rows, summaries
1348 | 
1349 | 
1350 | # ---------------------------------------------------------------------------
1351 | # CLI
1352 | # ---------------------------------------------------------------------------
1353 | 
1354 | 
1355 | def parse_args(argv: list[str]) -> argparse.Namespace:
1356 |     parser = argparse.ArgumentParser(
1357 |         description=(
1358 |             "Test whether preferences contain incremental information "
1359 |             "about future outcomes."
1360 |         )
1361 |     )
1362 |     parser.add_argument(
1363 |         "--track",
1364 |         choices=["synthetic", "arena"],
1365 |         default="synthetic",
1366 |     )
1367 |     parser.add_argument(
1368 |         "--seed",
1369 |         type=int,
1370 |         default=7,
1371 |         help="Fallback single seed when --seeds is omitted.",
1372 |     )
1373 |     parser.add_argument(
1374 |         "--seeds",
1375 |         default=None,
1376 |         help="Comma-separated run seeds, for example 1,2,3,4,5.",
1377 |     )
1378 |     parser.add_argument(
1379 |         "--test-fraction",
1380 |         type=float,
1381 |         default=0.2,
1382 |         help="Fraction of complete groups/sessions assigned to test.",
1383 |     )
1384 |     parser.add_argument(
1385 |         "--bootstrap-samples",
1386 |         type=int,
1387 |         default=2000,
1388 |         help="Hierarchical paired-bootstrap draws.",
1389 |     )
1390 |     parser.add_argument(
1391 |         "--bootstrap-seed",
1392 |         type=int,
1393 |         default=1729,
1394 |     )
1395 |     parser.add_argument(
1396 |         "--confidence-level",
1397 |         type=float,
1398 |         default=0.95,
1399 |     )
1400 |     parser.add_argument(
1401 |         "--limit-rows",
1402 |         type=int,
1403 |         default=None,
1404 |         help="Limit Hugging Face rows for an Arena smoke test.",
1405 |     )
1406 |     parser.add_argument(
1407 |         "--synthetic-sessions",
1408 |         type=int,
1409 |         default=5000,
1410 |     )
1411 |     parser.add_argument(
1412 |         "--synthetic-max-rounds",
1413 |         type=int,
1414 |         default=6,
1415 |     )
1416 |     parser.add_argument(
1417 |         "--synthetic-preference-effects",
1418 |         default="0.75",
1419 |         help=(
1420 |             "Comma-separated direct preference-to-future coefficients. "
1421 |             "Use 0 for the strict null."
1422 |         ),
1423 |     )
1424 |     parser.add_argument(
1425 |         "--synthetic-shared-latent-effect",
1426 |         type=float,
1427 |         default=0.0,
1428 |         help=(
1429 |             "Optional shared latent effect on the future. Leave at 0 for "
1430 |             "the strict null experiment."
1431 |         ),
1432 |     )
1433 |     parser.add_argument(
1434 |         "--out",
1435 |         default=None,
1436 |         help="CSV path for per-seed feature-set results.",
1437 |     )
1438 |     parser.add_argument(
1439 |         "--summary-out",
1440 |         default=None,
1441 |         help="CSV path for PFI/bootstrap summaries.",
1442 |     )
1443 |     return parser.parse_args(argv)
1444 | 
1445 | 
1446 | def main(argv: list[str]) -> int:
1447 |     args = parse_args(argv)
1448 | 
1449 |     if not 0.0 < args.test_fraction < 1.0:
1450 |         raise SystemExit("--test-fraction must be between 0 and 1.")
1451 |     if args.bootstrap_samples < 0:
1452 |         raise SystemExit("--bootstrap-samples must be non-negative.")
1453 | 
1454 |     seeds = parse_int_list(args.seeds, fallback=args.seed)
1455 |     for seed in seeds:
1456 |         seed_everything(seed)
1457 | 
1458 |     if args.track == "synthetic":
1459 |         rows, summaries = run_synthetic_matrix(args, seeds)
1460 |     elif args.track == "arena":
1461 |         rows, summaries = run_arena_matrix(args, seeds)
1462 |     else:
1463 |         raise AssertionError(args.track)
1464 | 
1465 |     print_header("PreferenceFutures final summary")
1466 |     summary_df = pd.DataFrame(
1467 |         [dataclasses.asdict(row) for row in summaries]
1468 |     )
1469 |     print(
1470 |         summary_df.to_string(
1471 |             index=False,
1472 |             float_format=lambda value: f"{value:.6f}",
1473 |         )
1474 |     )
1475 | 
1476 |     if args.out:
1477 |         detailed_df = pd.DataFrame(
1478 |             [dataclasses.asdict(row) for row in rows]
1479 |         )
1480 |         detailed_df.to_csv(args.out, index=False)
1481 |         print(f"\nSaved per-seed results to {args.out}")
1482 | 
1483 |     if args.summary_out:
1484 |         summary_df.to_csv(args.summary_out, index=False)
1485 |         print(f"Saved summary results to {args.summary_out}")
1486 | 
1487 |     return 0
1488 | 
1489 | 
1490 | if __name__ == "__main__":
1491 |     raise SystemExit(main(sys.argv[1:]))
```
