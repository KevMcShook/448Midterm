# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. Individual author: **Kevin Shook**.

This project compares a CNN and an LSTM on Claude, Gemini, and Qwen response attribution. Each architecture is trained with prompts only, responses only, and both together. R is used for data exploration; Python/PyTorch is used for modeling.

## Start here

- `report/project_report.pdf`: report with the measured results, methods, analysis, and limitations.
- `report/project_report.md`: editable report text.
- `explore_data.R`: RStudio exploration using the balanced 9,000-example dataset.
- `scripts/train_experiments.py`: required CNN and LSTM implementations and all six experiments.
- `scripts/analyze_results.py`: result figures, grouped bootstrap intervals, and RQ4 analysis.
- `results/all_metrics.json`: measured results and training histories.
- `data/cleaned/README.md`: data cleaning decisions and field definitions.

The existing `cleaning.R` loads the full cleaned data with `read_parquet()`. Open `448Midterm.Rproj` before running the R files. The installed Arrow build supports the cleaned files' gzip compression.

## Data

Download the seven original Parquet shards from [Arena Human Preference 140K](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k/tree/main/data) into `data/raw/`. The source dataset contains April-July 2025 text conversations. Source prompts are CC-BY-4.0; model outputs follow the respective providers' terms. The source card and cleaning documentation describe limitations. Raw files and large generated data are excluded from Git.

The balanced file contains 3,000 examples each from:

- `claude-opus-4-20250514`
- `gemini-2.5-flash`
- `qwen3-235b-a22b-no-thinking`

Required fields are `LLM_name`, `LLM_Input`, and `LLM_output`. Identifiers and model names are never classifier inputs. `split_group` joins related sessions, normalized identical prompts, and identical responses so they cannot cross data splits.

## Reproduce with Python 3.12

Create and activate a virtual environment. For a CPU machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If cleaned files are absent, generate them. The script refuses to overwrite existing cleaned data; use `--out data/regenerated` to make a comparison copy.

```bash
python scripts/clean_data.py
```

To verify implementation and reproduce the experiments, use a new output directory so the delivered results stay intact:

```bash
python -m unittest discover -s tests -v
python scripts/train_experiments.py --output results/reproduction
python scripts/analyze_results.py --results results/reproduction
python scripts/verify_results.py --results results/reproduction
python scripts/build_report.py --results results/reproduction --out report/reproduction
```

The default training configuration uses a shared approximately 70/15/15 grouped split, seed 448, a training-only vocabulary, 12 maximum epochs, validation macro-F1 checkpoint selection, and early stopping after three non-improving epochs. A matching interrupted run can be continued with `--resume`; completed experiment files are reused, while an incomplete experiment restarts. Stored checkpoints are local generated artifacts and are excluded from Git.

Results from a different software version, seed, or device can vary. Reported confidence intervals reflect test-group sampling uncertainty, not variation across training seeds. RQ4's formatting removal is a fixed-model robustness experiment, not a causal attribution method.

## Files to submit

Submit the GitHub repository URL with the code, results/figures, and PDF report available. The PDF names Kevin Shook as the individual author. Data can be recreated from the documented source and cleaning script; no private credentials or API calls are required. The source Parquet files exceed GitHub's ordinary per-file limit and should not be committed.

## Assistance and review

Codex assisted with data preparation, implementation, execution, verification, and drafting. The author should review the report and code and be able to explain the dataset decisions, models, evaluation, and limitations. The lessons section reports observations from these experiments rather than inventing personal experiences.
