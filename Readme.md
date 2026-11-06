# Name That LLM

Identify which LLM generated a paragraph of text. This is a supervised,
multiclass text classification project: `text` is the input and `model` is the
target label.

This repository currently contains project setup only. Data processing,
feature extraction, training, evaluation, and prediction are not implemented.

## Environment

Use Python 3.11 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

The environment includes pandas, scikit-learn, and PyArrow for future data work,
Ruff and pytest for development, and JupyterLab for exploration.

```sh
ruff check .
ruff format --check .
jupyter lab
```

The `tests/` directory is reserved for future tests; no tests are defined yet.
Run `pytest` once tests have been added.

## Layout

```text
data/
  raw/                  Original competition files; do not modify
  processed/            Future derived datasets
models/                 Future trained model artifacts
notebooks/              Future exploratory notebooks
reports/                Future evaluation results and figures
src/name_that_llm/      Python package; currently an empty scaffold
tests/                  Future automated tests
```

## Local competition data

The supplied competition files are copied from
`/Users/sahil/Downloads/name-that-llm/` into `data/raw/` during initial setup.
Datasets and generated artifacts are ignored by Git. On a fresh clone, place
your own competition files in `data/raw/`.

| File | Rows | Columns / purpose |
| --- | ---: | --- |
| `train.csv` | 2,699 | `id`, `prompt_id`, `text`, `model` |
| `test.csv` | 3 | `id`, `text` |
| `sample_submission.parquet` | — | Supplied submission template |

Keep `prompt_id` available for future evaluation splits so paragraphs from
the same prompt can be grouped to reduce data leakage. Confirm the required
submission schema against the supplied template before implementing exports.

The supplied starter notebook remains in the Downloads folder as reference;
it is not part of this setup.
