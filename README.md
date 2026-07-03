# BigSMILES MLM

A masked language model for polymer representations written in BigSMILES.

This repository fine-tunes
[`DeepChem/ChemBERTa-77M-MTR`](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR)
on a local corpus of 4,926,212 BigSMILES strings. The trained model and
tokenizer are included in `bigsmiles-mlm/`, so inference does not require
retraining.

## What is included

| Path | Purpose |
| --- | --- |
| `MLM.ipynb` | Loads the corpus, prepares the tokenizer, and trains the MLM |
| `big_smiles/` | 50 CSV shards containing the training corpus |
| `bigsmiles-mlm/` | Trained model and tokenizer for inference |
| `bigsmiles-chemberta-mlm/` | Retained checkpoints from the five-epoch training run |
| `runs/bigsmiles_mlm/` | TensorBoard logs |
| `train.csv`, `eval.csv` | Exported training and evaluation loss samples |

The GeoLDM, QM9, and plastic-mixture files are legacy experiments. They are
not required to train or use the BigSMILES MLM.

## Setup

The corpus CSV files use Git LFS. Pull them after cloning:

```bash
git clone https://github.com/Ahnd6474/BigsmilesMLM.git
cd BigsmilesMLM
git lfs pull
```

Create an environment and install the packages used by `MLM.ipynb`:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS and Linux:

```bash
source .venv/bin/activate
```

Then install the notebook and training dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch pandas transformers datasets accelerate jupyterlab tensorboard
```

`requirements.txt` also contains packages used by older experiments in this
repository. They are not needed for the MLM workflow.

## Use the trained model

The committed checkpoint can fill a masked position in a BigSMILES string:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_path = "bigsmiles-mlm"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForMaskedLM.from_pretrained(model_path)
model.eval()

text = "{[*]CC[MASK]CC[*]}"
inputs = tokenizer(text, return_tensors="pt")
mask_position = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[0, 1]

with torch.no_grad():
    logits = model(**inputs).logits[0, mask_position]

for token_id in logits.topk(5).indices:
    print(tokenizer.decode([token_id]))
```

With the committed checkpoint, this example returns `O`, `N`, `S`, `C`, and
`1` as the five highest-logit tokens.

## Train the model

Start JupyterLab and open `MLM.ipynb`:

```bash
jupyter lab
```

The notebook performs the following steps:

1. Reads every `big_smiles/polyBERT_len85_*.csv` shard and concatenates the
   `0` column.
2. Creates a 90/10 train-validation split with seed 42.
3. Loads the ChemBERTa tokenizer and adds `{`, `}`, `$`, and `[*]`.
4. Tokenizes each sequence to a maximum length of 256.
5. Applies dynamic masking with a 15% mask probability.
6. Fine-tunes for five epochs and writes TensorBoard logs and checkpoints.

The training configuration recorded in the notebook is:

| Setting | Value |
| --- | --- |
| Base model | `DeepChem/ChemBERTa-77M-MTR` |
| Corpus size | 4,926,212 strings |
| Validation fraction | 10% |
| Maximum sequence length | 256 tokens |
| Mask probability | 0.15 |
| Batch size | 32 per device |
| Epochs | 5 |
| Learning rate | `5e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.06` |

Full-corpus training is GPU-oriented and produced 769,725 optimizer steps in
the retained run. For a smoke test, load only one or two CSV shards before
starting the trainer.

## Monitor training

Read the saved event file with TensorBoard:

```bash
tensorboard --logdir runs/bigsmiles_mlm
```

The retained checkpoints are under `bigsmiles-chemberta-mlm/`. For normal
inference, use `bigsmiles-mlm/`; it contains the model, tokenizer,
vocabulary, and training arguments.

## Model details

The committed checkpoint is a RoBERTa masked-language model with:

- a 597-token vocabulary;
- 384 hidden dimensions;
- 3 hidden layers;
- 12 attention heads;
- a tokenizer limit of 512 positions.

The training notebook uses 256 positions even though the saved model supports
up to 512.

## License

See [LICENSE](LICENSE). The current license file retains the copyright and MIT
license notice from the GeoLDM source previously included in this repository.
