# Name That LLM

Identify which LLM generated a paragraph of text. This is a supervised,
multiclass text classification project: `text` is the input and `model` is the
target label.

The first goal is to learn the Transformer from *Attention Is All You Need* by
implementing its components in plain Python, then apply them to LLM attribution.
The current implementation covers tokenization, embedding lookup, single-head
and multi-head self-attention, residual addition, layer normalization, and the
position-wise feed-forward network, assembled into a reusable encoder stack. These are forward-pass calculations with
initialized parameters; there is no trained classifier yet.

## Environment

Use Python 3.11 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,notebooks]'
```

The environment includes pandas, scikit-learn, and PyArrow for data work,
Ruff and pytest for development, and JupyterLab for exploration. The Transformer
components use Python lists and standard-library modules; PyTorch is not required.

## Current components

| Component | Location | Behavior |
| --- | --- | --- |
| Tokenizer | `src/name_that_llm/tokenizer.py` | `tokenize`, `build_vocab`, and `encode`; preserves case and punctuation, with `<PAD>` and `<UNK>` IDs |
| Embeddings | `src/name_that_llm/embedding.py` | Reproducible random initialization and embedding lookup; padding row is zero |
| Sinusoidal positions | `notebooks/UnderstandingDataset.ipynb` | Creates positional vectors and adds them to token embeddings |
| Self-attention | `src/name_that_llm/attention.py` | Query/key/value projections, scaled dot-product scores, and stable softmax |
| Multi-head attention | `src/name_that_llm/attention.py` | Independent heads, concatenation, and output projection back to `d_model` |
| Residual addition | `src/name_that_llm/residual.py` | Elementwise addition with matching-shape checks |
| Layer normalization | `src/name_that_llm/normalization.py` | Per-token normalization across features, followed by scale and shift |
| Feed-forward network | `src/name_that_llm/feed_forward.py` | Two affine projections with ReLU between them, sharing parameters across token positions |
| Encoder stack | `src/name_that_llm/encoder.py` | Applies post-normalized encoder layers in order and returns the final token vectors and attention weights for each layer and head |

Attention functions receive projection matrices from the caller. Initialize
these once and reuse them for subsequent inputs. `multi_head_attention` returns
an output of shape `sequence_length × d_model` and weights of shape
`num_heads × sequence_length × sequence_length`.

Dropout, padding and causal masks, the Transformer decoder,
gradient computation, training, and classification evaluation are not
implemented yet.

## Notebook walkthrough

From the activated environment at the repository root:

```sh
jupyter lab notebooks/UnderstandingDataset.ipynb
```

Run the cells in order. The notebook explores the dataset and model counts,
then walks through vocabulary creation, embeddings, sinusoidal positions,
self-attention, residual addition, and a two-head attention example. It then
applies layer normalization and a feed-forward network that expands 8 features
to 32 and projects back to 8. A second residual connection and separate layer
normalization follow the feed-forward network. This demonstrates the two
encoder sublayers without dropout or masks. The final cells verify that an
`EncoderLayer` reproduces this calculation, then apply a second independently
initialized layer with `encoder_stack`. Parameters remain untrained.

The data-loading cell currently contains an absolute local path. Change
`data_dir` to your checkout's `data/raw/` directory when running elsewhere.

## Package example

The reusable functions can also be imported independently of the notebook:

```python
from name_that_llm.tokenizer import build_vocab, encode
from name_that_llm.embedding import initialize_embeddings, lookup_embeddings

training_texts = ["Hello, world!", "Hello again."]
vocab = build_vocab(training_texts)
E = initialize_embeddings(vocab, d=8, seed=42)

token_ids = encode("Hello, spaceship!", vocab)
vectors = lookup_embeddings(token_ids, E)

print(token_ids)
print("Embedding shape:", len(vectors), "x", len(vectors[0]))
```

Here, `spaceship` maps to `<UNK>`. Encoding keeps the vocabulary fixed. Build
the vocabulary from training texts only, then reuse it for validation, testing,
and future predictions.

## Checks

```sh
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

Tests cover tokenization, unknown-token handling, embedding initialization,
attention calculations, head concatenation and output projection, residual
addition, layer normalization, feed-forward projections and ReLU, and invalid
shapes.

## Layout

```text
data/
  raw/                  Original competition files; do not modify
  processed/            Future derived datasets
models/                 Future trained model artifacts
notebooks/              Dataset exploration and Transformer walkthrough
reports/                Future evaluation results and figures
src/name_that_llm/      Reusable Transformer components in plain Python
tests/                  Automated component tests
```

## Local competition data

Place the supplied competition files in `data/raw/`. Datasets and generated
artifacts are ignored by Git and must be supplied separately on a fresh clone.

| File | Rows | Columns / purpose |
| --- | ---: | --- |
| `train.csv` | 2,699 | `id`, `prompt_id`, `text`, `model` |
| `test.csv` | 3 | `id`, `text` |
| `sample_submission.parquet` | 3 | `id`, `model`; supplied submission template |

The notebook currently uses the labeled data for exploration. Before training
and evaluation, create train/validation/local-test partitions that keep each
`prompt_id` in one partition to reduce leakage. The supplied `test.csv` is the
separate competition prediction set. Confirm the submission schema against the
template before implementing exports.
