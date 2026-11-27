# Name That LLM

Given a paragraph of text, which LLM wrote it? This project builds the Transformer
encoder from *Attention Is All You Need* in plain Python and uses its output for multiclass text attribution.
The current forward pass produces one raw score per candidate model. The
encoder and classification head use randomly initialized weights. Attribution
accuracy will become meaningful once those weights have been trained and evaluated.

## Set up and run the walkthrough

1. From the repository root, create a Python 3.11+ environment and install the project.

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev,notebooks]'
   ```

2. Place the supplied competition files in `data/raw/`. Git ignores these files.
   The notebook reads the two CSVs. The Parquet file defines the competition submission format.

   | File | Rows | Columns |
   | --- | ---: | --- |
   | `train.csv` | 2,699 | `id`, `prompt_id`, `text`, `model` |
   | `test.csv` | 3 | `id`, `text` |
   | `sample_submission.parquet` | 3 | `id`, `model` |

3. Open the notebook and run all cells in order using the project environment.

   ```sh
   jupyter lab notebooks/UnderstandingDataset.ipynb
   ```

   The notebook inspects the data and counts examples per model, then traces
   `Hello, world!` through a two-layer encoder and prints a score for each model
   in the labeled CSV. The package example below also runs without competition data.

## How a paragraph gets a score

| Step | Calculation in the walkthrough | Package code in `src/name_that_llm/` |
| ---: | --- | --- |
| 1 | Split text into words and punctuation, preserving case. Build a vocabulary once, reserve IDs for `<PAD>` and `<UNK>`, and map each token to its ID. The example has four tokens. | `tokenizer.py` |
| 2 | Look up an eight-feature embedding for each ID. Add a sinusoidal position vector to each row so the encoder can distinguish token order. | `embedding.py` |
| 3 | Project the input into queries, keys, and values. Divide query–key dot products by the square root of the key width, apply row-wise softmax, then take a weighted sum of values for each token. | `attention.py` |
| 4 | Run two attention heads with independent projections, concatenate their results, and project back to eight features per token. | `attention.py` |
| 5 | Add the input to the attention output. Normalize each token across its features and apply the attention sublayer's scale and shift. | `residual.py`, `normalization.py` |
| 6 | Apply an affine projection from 8 to 32 features, ReLU, and a projection back to 8. Add the sublayer input and apply a second layer normalization with its own parameters. | `feed_forward.py`, `residual.py`, `normalization.py` |
| 7 | Verify that one encoder layer reproduces steps 4–6. Pass its output through a second independently initialized layer. The stack returns a 4 × 8 matrix and the attention weights for each layer and head. | `encoder.py` |
| 8 | Average each feature across the four token rows. This gives one eight-feature vector for the paragraph. | `classifier.py` |
| 9 | For each model label, take the dot product of its weight vector with the paragraph vector and add its bias. The label with the highest raw score (logit) is the forward-pass prediction. | `classifier.py` |

## Run a forward pass without the dataset

Run this example in the activated environment. Initialize the parameters once
and reuse the same vocabulary, embeddings, encoder layers, and head for each
paragraph. Supply one unpadded paragraph containing at least one token per call.

```python
import random

from name_that_llm.classifier import ClassifierHead, mean_pool
from name_that_llm.embedding import initialize_embeddings, lookup_embeddings, position_vector
from name_that_llm.encoder import EncoderLayer, encoder_stack
from name_that_llm.residual import residual_add
from name_that_llm.tokenizer import build_vocab, encode

training_texts = ["Hello, world!", "Hello again."]
labels = ["model-a", "model-b"]
d_model = 8
vocab = build_vocab(training_texts)
embeddings = initialize_embeddings(vocab, d=d_model, seed=42)
rng = random.Random(43)
layers = [EncoderLayer.initialize(d_model, 2, 32, rng) for _ in range(2)]
head = ClassifierHead.initialize(d_model, labels, rng)

token_ids = encode("Hello, spaceship!", vocab)  # spaceship maps to <UNK>
vectors = lookup_embeddings(token_ids, embeddings)
positions = [position_vector(i, d_model) for i in range(len(vectors))]
contextual, _ = encoder_stack(residual_add(vectors, positions), layers)
scores = head.forward(mean_pool(contextual))
print(scores)
print("Highest-scoring label (untrained):", max(scores, key=scores.get))
```

## Verify the implementation

```sh
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

## Repository layout

```text
data/raw/             Original competition files
data/processed/       Derived datasets
models/               Trained model artifacts
notebooks/            Dataset exploration and Transformer walkthrough
reports/              Evaluation results and figures
src/name_that_llm/    Reusable components
tests/                Automated checks
```

## Next steps for training

1. Split the labeled rows into training, validation, and local test partitions.
   Keep every `prompt_id` in one partition so responses to the same prompt stay together.
2. Build the vocabulary and label mapping from the training partition. Reuse
   them for validation, local testing, and competition predictions.
3. Implement the training loss, gradients, and parameter updates. The current
   encoder processes individual unpadded sequences. Padded batches would also
   need masking in attention and pooling.
4. Select the model using validation results, evaluate on the held-out local
   test partition, then predict labels for the separate competition `test.csv`.
   Match the IDs and schema in `sample_submission.parquet` when exporting.
