# Curated knowledge corpus

Only manually approved sources enter an index. `model/` contains project-owned, reviewable model documentation. `approved-external-sources.json` records authoritative external candidates, but a candidate is not indexable until a reviewer creates a local snapshot, records its exact SHA-256 in `sources.json`, and confirms the licence note.

Raw external files belong under `corpus/raw/`; that directory and all generated indexes are Git-ignored. This prevents a URL changing silently and avoids committing third-party full text. The production source manifest must use the strict fields validated by `SourceRecord`, including `local_path`, `sha256`, and `ingested_at`.

Run `python PythonServices/TreeSemKnowledge/ingestion/fetch_sources.py --manifest PythonServices/TreeSemKnowledge/corpus/sources.json --source-root PythonServices/TreeSemKnowledge/corpus` to fetch registered snapshots. If a publisher changes a page, the checksum mismatch stops ingestion until a human reviews the new content and deliberately updates the manifest.

The current official-source review used the WHO 2025 consolidated PPH guideline, the RCOG patient page, and the NHS early-days page. The RCOG page is explicitly marked as under review, so its snapshot must not be treated as newer professional guidance.
