# GeoLDM: Geometric Latent Diffusion Models for 3D Molecule Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/GeoLDM/blob/main/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2305.01140-B31B1B.svg)](https://arxiv.org/abs/2305.01140)

<!-- [[Code](https://github.com/MinkaiXu/GeoLDM)] -->

![cover](equivariant_diffusion/framework.png)

Official code release for the paper "Geometric Latent Diffusion Models for 3D Molecule Generation", accepted at *International Conference on Machine Learning, 2023*.

## Environment

Install the required packages from `requirements.txt`. A simplified version of the requirements can be found [here](https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/requirements.txt).

**Note**: If you want to set up a rdkit environment, it may be easiest to install conda and run:
``conda create -c conda-forge -n my-rdkit-env rdkit`` and then install the other required packages. But the code should still run without rdkit installed though.

## Usage

The repository is organized around training scripts (`main_qm9.py`, `main_geom_drugs.py`), evaluation utilities (for example `eval_sample.py` and `eval_analyze.py`), and helper modules in the `qm9/` and `equivariant_diffusion/` packages. Typical workflows involve the following steps:

1. **Prepare the environment** by installing the dependencies listed above.
2. **Download the dataset** (for Drugs, follow the extra steps under `data/geom/README.md`).
3. **Train or download** a pretrained GeoLDM checkpoint (see the sections below).
4. **Run the evaluation utilities** or load the checkpoint programmatically in Python for custom experiments.

### Command line quickstart

Once you have placed a pretrained model in `outputs/$exp_name` (or finished training one), you can analyze or visualize generated molecules with:

```bash
python eval_analyze.py --model_path outputs/$exp_name --n_samples 10_000
python eval_sample.py --model_path outputs/$exp_name --n_samples 10_000
```

Both commands will read the stored training configuration from `outputs/$exp_name/args.pickle`, rebuild the latent diffusion model, and write results to `outputs/$exp_name/eval/`.

### Python API example

You can also access the latent diffusion model directly from Python to integrate GeoLDM into custom pipelines. The following example loads a pretrained QM9 checkpoint, draws a batch of molecules, and prints their shapes:

```python
import pickle
from os.path import join

import torch

from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_latent_diffusion
from qm9.sampling import sample

model_dir = "outputs/geoldm_qm9"  # folder containing args.pickle and weights

with open(join(model_dir, "args.pickle"), "rb") as f:
    train_args = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_args.device = device
train_args.cuda = device.type == "cuda"

dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h)
dataloaders, _ = dataset.retrieve_dataloaders(train_args)

ldm, nodes_dist, _ = get_latent_diffusion(
    train_args, device, dataset_info, dataloaders["train"],
)

weights = torch.load(join(model_dir, "generative_model_ema.npy"), map_location=device)
ldm.load_state_dict(weights)
ldm.eval()

one_hot, charges, positions, node_mask = sample(
    train_args, device, ldm, dataset_info, nodesxsample=nodes_dist.sample(4)
)

print(one_hot.shape, charges.shape, positions.shape, node_mask.shape)
```

The snippet mirrors the logic in `eval_sample.py`: it restores the configuration, reconstructs the model and data utilities, and then uses `qm9.sampling.sample` to draw molecules. Replace `model_dir` with the path to your own experiment when working with different checkpoints.

### `qm9.model_module` helper utilities

For higher-level workflows you can rely on the convenience wrappers bundled in [`qm9/model_module.py`](qm9/model_module.py). The module re-exports helpers to load checkpoints, encode/decode molecules and run latent diffusion sampling without having to piece the pieces together manually. A minimal session looks as follows:

```python
import pickle
from pathlib import Path

import torch

from configs.datasets_config import get_dataset_info
from qm9 import dataset, model_module

model_dir = Path("outputs/geoldm_qm9")

with model_dir.joinpath("args.pickle").open("rb") as f:
    train_args = pickle.load(f)

dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h)
dataloaders, _ = dataset.retrieve_dataloaders(train_args)

ldm, nodes_dist, _ = model_module.load_model(
    stage="latent_diffusion",
    args=train_args,
    dataset_info=dataset_info,
    dataloader_train=dataloaders["train"],
    checkpoint_path=model_dir,
)

batch = next(iter(dataloaders["train"]))
device = next(ldm.parameters()).device

x = batch["positions"].to(device)
h = {
    "categorical": batch["one_hot"].to(device),
    "integer": batch["charges"].to(device),
}
node_mask = batch["atom_mask"].unsqueeze(-1).to(device)
edge_mask = batch["edge_mask"].to(device)

latents = model_module.encode(ldm, x, h, node_mask=node_mask, edge_mask=edge_mask)
recon_positions, recon_features = model_module.decode(
    ldm, latents[0], node_mask=node_mask, edge_mask=edge_mask
)

conformer = model_module.smiles_to_3d("CCO")[0]
smiles, _ = model_module.structure_to_smiles(
    conformer["atom_symbols"], conformer["coordinates"]
)
```

`load_model` reconstructs the requested stage (diffusion, autoencoder or latent diffusion) and automatically locates the EMA weights saved by the training scripts. The combination of `encode`/`decode` lets you round-trip batches from the dataloader, while `smiles_to_3d` and `structure_to_smiles` bridge between string and 3D representations. `run_diffusion` exposes the raw sampling loop used internally by `qm9.sampling.sample`—refer to that helper for an end-to-end example that prepares the required node/edge masks.


## Train the GeoLDM

### For QM9

```python main_qm9.py --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 64 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9```

### For Drugs

First follow the intructions at `data/geom/README.md` to set up the data.

```python main_geom_drugs.py --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000 --train_diffusion --trainable_ae --latent_nf 2 --exp_name geoldm_drugs```

**Note**: In the paper, we present an encoder early-stopping strategy for training the Autoencoder. However, in later experiments, we found that we can even just keep the encoder untrained and only train the decoder, which is faster and leads to similar results. Our released version uses this strategy. This phenomenon is quite interesting and we are also still actively investigating it.

### Pretrained models

We also provide pretrained models for both QM9 and Drugs. You can download them from [here](https://drive.google.com/drive/folders/1EQ9koVx-GA98kaKBS8MZ_jJ8g4YhdKsL?usp=sharing). The pretrained models are trained with the same hyperparameters as the above commands except that latent dimensions `--latent_nf` are set as 2 (the results should be roughly the same if as 1). You can load them for running the following evaluations by putting them in the `outputs` folder and setting the argument `--model_path` to the path of the pretrained model `outputs/$exp_name`.

## Evaluate the GeoLDM

To analyze the sample quality of molecules:

```python eval_analyze.py --model_path outputs/$exp_name --n_samples 10_000```

To visualize some molecules:

```python eval_sample.py --model_path outputs/$exp_name --n_samples 10_000```

Small note: The GPUs used for these experiment were pretty large. If you run out of GPU memory, try running at a smaller size.
<!-- The main reason is that the EGNN runs with fully connected message passing, which becomes very memory intensive. -->

## Conditional Generation

### Train the Conditional GeoLDM

```python main_qm9.py --exp_name exp_cond_alpha  --model egnn_dynamics --lr 1e-4  --nf 192 --n_layers 9 --save_model True --diffusion_steps 1000 --sin_embedding False --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --dequantization deterministic --include_charges False --diffusion_loss_type l2 --batch_size 64 --normalize_factors [1,8,1] --conditioning alpha --dataset qm9_second_half --train_diffusion --trainable_ae --latent_nf 1```

The argument `--conditioning alpha` can be set to any of the following properties: `alpha`, `gap`, `homo`, `lumo`, `mu` `Cv`. The same applies to the following commands that also depend on alpha.

### Generate samples for different property values

```python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --property alpha --n_sweeps 10 --task qualitative```

### Evaluate the Conditional GeoLDM with property classifiers

#### Train a property classifier
```cd qm9/property_prediction```  
```python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha --model_name egnn```

Additionally, you can change the argument `--model_name egnn` by `--model_name numnodes` to train a classifier baseline that classifies only based on the number of nodes.

#### Evaluate the generated samples

Evaluate the trained property classifier on the samples generated by the trained conditional GeoLDM model

```python eval_conditional_qm9.py --generators_path outputs/exp_cond_alpha --classifiers_path qm9/property_prediction/outputs/exp_class_alpha --property alpha  --iterations 100  --batch_size 100 --task edm```

## Citation
Please consider citing the our paper if you find it helpful. Thank you!
```
@inproceedings{xu2023geometric,
  title={Geometric Latent Diffusion Models for 3D Molecule Generation},
  author={Minkai Xu and Alexander Powers and Ron Dror and Stefano Ermon and Jure Leskovec},
  booktitle={International Conference on Machine Learning},
  year={2023},
  organization={PMLR}
}
```

## Acknowledgements

This repo is built upon the previous work [EDM](https://arxiv.org/abs/2203.17003). Thanks to the authors for their great work!