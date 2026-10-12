# Cleaned Claude, Gemini, and Qwen responses

Source: [Arena Human Preference 140K](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k). Prepared from all seven downloaded shards in `../raw`; their SHA-256 hashes were checked before and after processing and did not change.

## Files and counts

`claude_gemini_qwen_balanced.parquet` is the suggested starting dataset: 9,000 examples, with one exact model version per family and 3,000 examples per family.

`claude_gemini_qwen_all.parquet` retains all 51,740 cleaned examples across the selected families' model versions. This larger collection is not balanced. These two files overlap; do not concatenate them or treat them as separate training and test sets.

| Family | All cleaned | Balanced subset | Model used in balanced subset |
| --- | ---: | ---: | --- |
| Claude | 23,996 | 3,000 | claude-opus-4-20250514 |
| Gemini | 17,269 | 3,000 | gemini-2.5-flash |
| Qwen | 10,475 | 3,000 | qwen3-235b-a22b-no-thinking |

`cleaning_report.json` records all filter counts, model counts, source hashes, output hashes, and validation results.

## Cleaning decisions

- Read 135,634 comparisons; retain source language `en` and `evaluation_order == 1`. Language labels are inherited, not independently reclassified.
- Match model names starting with `claude-`, `gemini-`, or `qwen`. `qwq-32b` is excluded rather than silently broadening the requested Qwen prefix to aliases.
- For each eligible side, use the first user message and its immediately following assistant response. Do not combine sides or use `full_conversation`, whose model identities can change across evaluation orders.
- Keep only plain text content with a nonempty prompt and response. Remove obvious error-only responses using a narrow whole-response pattern.
- Normalize Unicode to NFC, line endings to LF, remove NUL characters, and trim outer whitespace. Preserve casing, punctuation, Markdown, code, and internal whitespace. No stemming, stop-word removal, tokenization, or length truncation has been applied.
- Remove responses matching narrow explicit self-identification patterns such as “I am Claude” or “developed by Google.” This is a heuristic that may also match quoted statements; it is not exhaustive removal of all model-name mentions.
- Deduplicate by exact model version, normalized prompt (case-insensitive and whitespace-normalized), and normalized response (case-sensitive and whitespace-normalized). Different versions and different models answering the same prompt remain separate examples.
- Choose the three balanced-subset model versions listed above from the available counts; use deterministic SHA-256 ordering with seed label 448 to select 3,000 examples per version. This controls class counts, not topic or date balance.
- Keep all preference outcomes; the winner is not the prediction target and is omitted from the cleaned files.

Removed candidate responses (sequential, non-overlapping counts):
- explicit identity response: 366
- non text or malformed content: 15
- duplicate model prompt response: 58

## Fields and use

- Required assignment fields: `LLM_name` (lowercase family label), `LLM_Input` (prompt), and `LLM_output` (response).
- Exact model and provenance: `model_version`, `example_id`, `source_file`, `source_id`, `source_side`, `evaluation_session_id`, `evaluation_order`, and `timestamp` (source timestamp preserved without asserting a timezone).
- Split metadata: `prompt_id`, `response_id`, and `split_group`.
- Optional analysis: `language`, `is_code`, `is_math`, `is_creative_writing`, and whitespace-based prompt/response word counts. Category labels come from the source and may be null or overlap.

Split by **`split_group`**, not by individual rows. Connected examples sharing a normalized prompt, exact normalized response, or evaluation session receive the same group. There are 38,186 groups in the full cleaned data; the largest contains 57 examples. Near-duplicate paraphrases are not detected. No train/validation/test split has been assigned.

Only pass the requested prompt and/or response text to the classifiers. Do not include model names, IDs, source metadata, or analysis tags as input features. Build vocabulary and fit preprocessing on training data only.

These records come from naturally occurring prompts, so topic, date, model settings, and response length may correlate with family. The balanced subset represents three specific model configurations, not every model in each family. Poor or strong prompt-only performance should be interpreted in that context.

## Source terms and limitations

The source card licenses user prompts under CC-BY-4.0 and places model outputs under the respective providers' terms. Cite the original source and document this transformation. Public user conversations may contain sensitive material; this preparation is structural cleaning and is not a comprehensive privacy or content audit. The raw data and project code were not changed.

## R compatibility

Both Parquet files use gzip compression, supported by the installed R Arrow build. Re-encoding from Zstandard was checked to preserve every value and row order. Read these files with `arrow::read_parquet()`, optionally wrapped in `data.table::as.data.table()`. `fread()` does not read Parquet.
