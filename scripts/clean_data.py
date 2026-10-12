"""Reproduce the cleaning without changing raw files. Requires PyArrow."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
import unicodedata
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import argparse
parser = argparse.ArgumentParser(description='Reproduce the documented Arena cleaning and balanced sampling.')
parser.add_argument('--raw', type=Path, default=Path('data/raw'))
parser.add_argument('--out', type=Path, default=Path('data/cleaned'))
args = parser.parse_args()
RAW, OUT = args.raw, args.out
if any((OUT / name).exists() for name in ['claude_gemini_qwen_all.parquet', 'claude_gemini_qwen_balanced.parquet']):
    raise SystemExit('Output data exists. Choose a fresh --out directory to avoid replacing it.')
SUMMARY = OUT / 'README.md'
MODELS = {'claude': 'claude-opus-4-20250514', 'gemini': 'gemini-2.5-flash', 'qwen': 'qwen3-235b-a22b-no-thinking'}
SOURCE = 'https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k'
counts = Counter()
raw_models = Counter()
rows = []
seen = set()
manifest = []

def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def clean_text(text):
    return unicodedata.normalize('NFC', text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')).strip()

def match_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def message_text(msg):
    parts = msg.get('content')
    if not isinstance(parts, list) or not parts:
        return None
    if any(p.get('type') != 'text' or p.get('image') or not isinstance(p.get('text'), str) for p in parts):
        return None
    return clean_text('\n'.join(p['text'] for p in parts))

identity = re.compile(r"\b(?:i\s*(?:am|'m|’m)|my name is|as)\s+(?:an?\s+)?(?:claude|gemini|qwen|chatgpt)\b|\b(?:developed|created|trained|built|made)\s+by\s+(?:anthropic|google|alibaba|qwen|openai)\b", re.I)
failure = re.compile(r'^(?:\*\*)?(?:internal server error|network error|request timed out|rate limit exceeded|service unavailable|error generating (?:a )?response|something went wrong)(?:\*\*)?[.!\s]*$', re.I)

paths = sorted(RAW.glob('*.parquet'))
assert len(paths) == 7, f'Expected 7 source shards, found {len(paths)}'
for path in paths:
    f = pq.ParquetFile(path)
    manifest.append({'file': path.name, 'bytes': path.stat().st_size, 'rows': f.metadata.num_rows, 'sha256': file_hash(path)})
    cols = ['id','model_a','model_b','evaluation_session_id','evaluation_order','conversation_a','conversation_b','category_tag','language','is_code','timestamp']
    for batch in f.iter_batches(batch_size=512, columns=cols):
        table = pa.Table.from_batches([batch])
        counts['source_comparisons'] += table.num_rows
        english = pc.equal(table['language'], 'en')
        counts['non_english_or_unknown_comparisons'] += table.num_rows - int(pc.sum(pc.cast(english, pa.int64())).as_py() or 0)
        table = table.filter(english)
        first = pc.equal(table['evaluation_order'], 1)
        counts['later_evaluation_english_comparisons'] += table.num_rows - int(pc.sum(pc.cast(first, pa.int64())).as_py() or 0)
        table = table.filter(first)
        counts['english_first_evaluation_comparisons'] += table.num_rows
        target = pc.or_(pc.match_substring_regex(table['model_a'], '^(claude-|gemini-|qwen)'), pc.match_substring_regex(table['model_b'], '^(claude-|gemini-|qwen)'))
        table = table.filter(target)
        counts['comparisons_with_requested_family'] += table.num_rows
        for source in table.to_pylist():
            for side in ('a','b'):
                model = source['model_'+side]
                family = next((n for n in MODELS if model.startswith(n)), None)
                if family is None:
                    continue
                counts['candidate_responses'] += 1
                raw_models[model] += 1
                conv = source['conversation_'+side]
                # Only use an unambiguous first user/assistant pair, not mixed history.
                if not isinstance(conv, list) or len(conv) < 2 or [m.get('role') for m in conv[:2]] != ['user','assistant']:
                    counts['removed_unexpected_conversation_structure'] += 1
                    continue
                prompt, response = message_text(conv[0]), message_text(conv[1])
                if prompt is None or response is None:
                    counts['removed_non_text_or_malformed_content'] += 1
                    continue
                if not prompt or not response:
                    counts['removed_empty_prompt_or_response'] += 1
                    continue
                if failure.fullmatch(response):
                    counts['removed_error_only_response'] += 1
                    continue
                if identity.search(response):
                    counts['removed_explicit_identity_response'] += 1
                    continue
                prompt_id = digest(match_text(prompt).casefold())
                response_id = digest(match_text(response))
                key = (model, prompt_id, response_id)
                if key in seen:
                    counts['removed_duplicate_model_prompt_response'] += 1
                    continue
                seen.add(key)
                tags = source.get('category_tag') or {}
                timestamp = source['timestamp'].isoformat() if source['timestamp'] else None
                rows.append({
                    'example_id': digest(json.dumps(key, separators=(',',':'))),
                    'LLM_name': family,
                    'LLM_Input': prompt,
                    'LLM_output': response,
                    'model_version': model,
                    'prompt_id': prompt_id,
                    'response_id': response_id,
                    'split_group': '',
                    'source_id': source['id'],
                    'evaluation_session_id': source['evaluation_session_id'],
                    'source_side': side,
                    'source_file': path.name,
                    'evaluation_order': source['evaluation_order'],
                    'language': source['language'],
                    'timestamp': timestamp,
                    'is_code': source['is_code'],
                    'is_math': (tags.get('math_v0.1') or {}).get('math'),
                    'is_creative_writing': (tags.get('creative_writing_v0.1') or {}).get('creative_writing'),
                    'prompt_word_count': len(prompt.split()),
                    'response_word_count': len(response.split()),
                })
    print(json.dumps({'completed_shard': path.name, 'retained_so_far': len(rows)}), flush=True)

# Connected groups prevent the same prompt, exact response, or session crossing splits.
parents = list(range(len(rows)))
def root(i):
    while parents[i] != i:
        parents[i] = parents[parents[i]]
        i = parents[i]
    return i

def union(i,j):
    a,b = root(i),root(j)
    if a != b:
        parents[max(a,b)] = min(a,b)

keys = {}
for i,row in enumerate(rows):
    for col in ('prompt_id','response_id','evaluation_session_id'):
        value = row[col]
        if value:
            key = (col, value)
            if key in keys:
                union(i, keys[key])
            else:
                keys[key] = i
for i,row in enumerate(rows):
    row['split_group'] = rows[root(i)]['example_id']

family_counts = Counter(r['LLM_name'] for r in rows)
model_counts = Counter(r['model_version'] for r in rows)
target_size = min(3000, *(model_counts[m] for m in MODELS.values()))
assert target_size >= 1000, model_counts
balanced = []
for family, model in MODELS.items():
    candidates = [r for r in rows if r['model_version'] == model]
    candidates.sort(key=lambda r: digest('seed=448|'+r['example_id']))
    balanced.extend(candidates[:target_size])
balanced.sort(key=lambda r: digest('order=448|'+r['example_id']))
assert Counter(r['LLM_name'] for r in balanced) == {k: target_size for k in MODELS}

OUT.mkdir(parents=True, exist_ok=True)
outputs = []
for name, records in [('claude_gemini_qwen_all.parquet',rows), ('claude_gemini_qwen_balanced.parquet',balanced)]:
    path = OUT / name
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite {path}')
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression='gzip', compression_level=6)
    restored = pq.read_table(path)
    assert restored.equals(table)
    assert len(set(table['example_id'].to_pylist())) == len(records)
    assert set(table['LLM_name'].to_pylist()) == set(MODELS)
    for col in ('LLM_Input','LLM_output','model_version','split_group'):
        assert not table[col].null_count
    outputs.append({'file':name,'rows':len(records),'bytes':path.stat().st_size,'sha256':file_hash(path)})

for item,path in zip(manifest, paths):
    assert file_hash(path) == item['sha256'], f'Source changed: {path}'

group_sizes = Counter(r['split_group'] for r in rows)
report = {
    'created_utc':datetime.now(timezone.utc).isoformat(), 'source_url':SOURCE,
    'counts':dict(counts), 'cleaned_family_counts':dict(family_counts),
    'candidate_model_counts':dict(raw_models), 'cleaned_model_counts':dict(model_counts),
    'balanced_model_versions':MODELS, 'balanced_per_family':target_size,
    'balanced_total':len(balanced), 'selection':'SHA-256 order of seed=448|example_id, first N per chosen model',
    'split_groups':len(group_sizes), 'largest_split_group':max(group_sizes.values()),
    'source_files':manifest, 'outputs':outputs,
    'validation':{'parquet_round_trip':True,'unique_example_ids':True,'balanced_counts':True,'original_sha256_unchanged':True},
}
assert counts['candidate_responses'] == len(rows) + sum(v for k,v in counts.items() if k.startswith('removed_'))
(OUT/'cleaning_report.json').write_text(json.dumps(report, indent=2)+'\n')
family_lines = '\n'.join(f'| {k.title()} | {family_counts[k]:,} | {target_size:,} | {MODELS[k]} |' for k in MODELS)
removal_lines = '\n'.join(f'- {k.removeprefix("removed_").replace("_"," ")}: {v:,}' for k,v in counts.items() if k.startswith('removed_'))
readme = f'''# Cleaned Claude, Gemini, and Qwen responses

Source: [Arena Human Preference 140K]({SOURCE}). Prepared from all seven downloaded shards in `../raw`; their SHA-256 hashes were checked before and after processing and did not change.

## Files and counts

`claude_gemini_qwen_balanced.parquet` is the suggested starting dataset: {len(balanced):,} examples, with one exact model version per family and {target_size:,} examples per family.

`claude_gemini_qwen_all.parquet` retains all {len(rows):,} cleaned examples across the selected families' model versions. This larger collection is not balanced. These two files overlap; do not concatenate them or treat them as separate training and test sets.

| Family | All cleaned | Balanced subset | Model used in balanced subset |
| --- | ---: | ---: | --- |
{family_lines}

`cleaning_report.json` records all filter counts, model counts, source hashes, output hashes, and validation results.

## Cleaning decisions

- Read {counts['source_comparisons']:,} comparisons; retain source language `en` and `evaluation_order == 1`. Language labels are inherited, not independently reclassified.
- Match model names starting with `claude-`, `gemini-`, or `qwen`. `qwq-32b` is excluded rather than silently broadening the requested Qwen prefix to aliases.
- For each eligible side, use the first user message and its immediately following assistant response. Do not combine sides or use `full_conversation`, whose model identities can change across evaluation orders.
- Keep only plain text content with a nonempty prompt and response. Remove obvious error-only responses using a narrow whole-response pattern.
- Normalize Unicode to NFC, line endings to LF, remove NUL characters, and trim outer whitespace. Preserve casing, punctuation, Markdown, code, and internal whitespace. No stemming, stop-word removal, tokenization, or length truncation has been applied.
- Remove responses matching narrow explicit self-identification patterns such as “I am Claude” or “developed by Google.” This is a heuristic that may also match quoted statements; it is not exhaustive removal of all model-name mentions.
- Deduplicate by exact model version, normalized prompt (case-insensitive and whitespace-normalized), and normalized response (case-sensitive and whitespace-normalized). Different versions and different models answering the same prompt remain separate examples.
- Choose the three balanced-subset model versions listed above from the available counts; use deterministic SHA-256 ordering with seed label 448 to select {target_size:,} examples per version. This controls class counts, not topic or date balance.
- Keep all preference outcomes; the winner is not the prediction target and is omitted from the cleaned files.

Removed candidate responses (sequential, non-overlapping counts):
{removal_lines}

## Fields and use

- Required assignment fields: `LLM_name` (lowercase family label), `LLM_Input` (prompt), and `LLM_output` (response).
- Exact model and provenance: `model_version`, `example_id`, `source_file`, `source_id`, `source_side`, `evaluation_session_id`, `evaluation_order`, and `timestamp` (source timestamp preserved without asserting a timezone).
- Split metadata: `prompt_id`, `response_id`, and `split_group`.
- Optional analysis: `language`, `is_code`, `is_math`, `is_creative_writing`, and whitespace-based prompt/response word counts. Category labels come from the source and may be null or overlap.

Split by **`split_group`**, not by individual rows. Connected examples sharing a normalized prompt, exact normalized response, or evaluation session receive the same group. There are {len(group_sizes):,} groups in the full cleaned data; the largest contains {max(group_sizes.values()):,} examples. Near-duplicate paraphrases are not detected. No train/validation/test split has been assigned.

Only pass the requested prompt and/or response text to the classifiers. Do not include model names, IDs, source metadata, or analysis tags as input features. Build vocabulary and fit preprocessing on training data only.

These records come from naturally occurring prompts, so topic, date, model settings, and response length may correlate with family. The balanced subset represents three specific model configurations, not every model in each family. Poor or strong prompt-only performance should be interpreted in that context.

## Source terms and limitations

The source card licenses user prompts under CC-BY-4.0 and places model outputs under the respective providers' terms. Cite the original source and document this transformation. Public user conversations may contain sensitive material; this preparation is structural cleaning and is not a comprehensive privacy or content audit. The raw data and project code were not changed.
'''
(OUT/'README.md').write_text(readme)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.write_text(readme)
print(json.dumps({'cleaned_family_counts':dict(family_counts),'balanced_per_family':target_size,'balanced_total':len(balanced),'counts':dict(counts),'split_groups':len(group_sizes),'largest_split_group':max(group_sizes.values()),'outputs':outputs}, indent=2), flush=True)
