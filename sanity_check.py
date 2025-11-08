from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import torch

from geoldm.configs import get_dataset_info
from geoldm.qm9 import dataset, load_model, sampling, visualize_molecule_3d


def _checkpoint_available(checkpoint_dir: Path) -> bool:
    required_files: Iterable[str] = (
        "args.pickle",
        "generative_model_ema.npy",
        "generative_model.npy",
    )
    return any((checkpoint_dir / filename).exists() for filename in required_files)


def main() -> None:
    checkpoint_dir = Path("outputs/geoldm_qm9")
    args_path = checkpoint_dir / "args.pickle"

    if not args_path.exists():
        print(
            "Skipping GeoLDM sanity check: training arguments were not found at"
            f" {args_path!s}."
        )
        print(
            "Provide pretrained artefacts in 'outputs/geoldm_qm9/' to run the full"
            " integration test."
        )
        return

    if not _checkpoint_available(checkpoint_dir):
        print(
            "Skipping GeoLDM sanity check: checkpoint weights are missing from"
            f" {checkpoint_dir!s}."
        )
        print("Expected 'generative_model_ema.npy' or 'generative_model.npy'.")
        return

    with args_path.open("rb") as f:
        train_args = pickle.load(f)

    dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h)
    dataloaders, _ = dataset.retrieve_dataloaders(train_args)
    train_loader = dataloaders["train"]

    ldm, nodes_dist, _ = load_model(
        stage="latent_diffusion",
        args=train_args,
        dataset_info=dataset_info,
        dataloader_train=train_loader,
        checkpoint_path=checkpoint_dir,
    )

    batch = next(iter(train_loader))
    device = next(ldm.parameters()).device
    node_mask = batch["atom_mask"].unsqueeze(-1).to(device)
    edge_mask = batch["edge_mask"].to(device)

    samples = sampling.sample(
        train_args,
        device,
        ldm,
        dataset_info,
        nodesxsample=nodes_dist.sample(4),
        prop_dist=None,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

    positions, features = samples[0], samples[1]
    one_hot = features["categorical"][0].cpu().numpy()
    atom_types = one_hot.argmax(axis=-1)
    atom_symbols = [dataset_info["atom_decoder"][i] for i in atom_types]
    visualize_molecule_3d(atom_symbols, positions[0].cpu().numpy())


if __name__ == "__main__":
    main()
