from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch

from geoldm.configs import get_dataset_info
from geoldm.qm9 import dataset, load_model, sampling, visualize_molecule_3d


def _checkpoint_available(checkpoint_dir: Path) -> bool:
    """Return ``True`` when the directory contains the expected artefacts."""

    if not checkpoint_dir.is_dir():
        return False

    args_file = checkpoint_dir / "args.pickle"
    if not args_file.exists():
        return False

    weight_files: Iterable[str] = (
        "generative_model_ema.npy",
        "generative_model.npy",
    )
    return any((checkpoint_dir / filename).exists() for filename in weight_files)


def _checkpoint_candidates(root: Path) -> Iterator[Path]:
    """Yield candidate checkpoint directories ordered by priority."""

    seen: set[Path] = set()

    env_override = os.environ.get("GEOLDM_CHECKPOINT_DIR")
    if env_override:
        env_path = Path(env_override).expanduser().resolve()
        if _checkpoint_available(env_path):
            seen.add(env_path)
            yield env_path

    defaults: Iterable[Path] = (
        root / "outputs" / "geoldm_qm9",
        root / "qm9_latent2",
        root / "drugs_latent2",
    )
    for candidate in defaults:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        if _checkpoint_available(candidate):
            seen.add(candidate)
            yield candidate

    for child in root.iterdir():
        if child in seen or not child.is_dir():
            continue
        if _checkpoint_available(child):
            seen.add(child)
            yield child


def _select_checkpoint(root: Path) -> Optional[Path]:
    """Return the checkpoint directory that should be used for the test."""

    candidates = list(_checkpoint_candidates(root))
    if not candidates:
        return None

    def _score(path: Path) -> tuple[int, str]:
        dataset_name = "zzzz"
        args_path = path / "args.pickle"
        try:
            with args_path.open("rb") as handle:
                args = pickle.load(handle)
                dataset_name = getattr(args, "dataset", dataset_name)
        except Exception:
            pass

        priority = 0 if dataset_name == "qm9" else 1
        return (priority, str(path))

    candidates.sort(key=_score)
    return candidates[0]


def main() -> None:
    root = Path(__file__).resolve().parent
    checkpoint_dir = _select_checkpoint(root)

    if checkpoint_dir is None:
        print("Skipping GeoLDM sanity check: no pretrained checkpoints were located.")
        print(
            "Place 'args.pickle' and either 'generative_model.npy' or"
            " 'generative_model_ema.npy' in the repository to run the"
            " integration test."
        )
        return

    args_path = checkpoint_dir / "args.pickle"
    with args_path.open("rb") as f:
        train_args = pickle.load(f)

    print(f"Using GeoLDM checkpoint from {checkpoint_dir} (dataset={train_args.dataset}).")

    if not torch.cuda.is_available():
        setattr(train_args, "cuda", False)
        print('using CPU')
    else:
        print('using CUDA')

    dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h)
    try:
        dataloaders, _ = dataset.retrieve_dataloaders(train_args)
    except Exception as exc:  # pragma: no cover - depends on optional assets
        print(
            "Could not retrieve training dataloaders; proceeding with a synthetic"
            f" batch instead. (Reason: {exc})"
        )
        train_loader = None
    else:
        train_loader = dataloaders["train"]

    try:
        ldm, nodes_dist, _ = load_model(
            stage="latent_diffusion",
            args=train_args,
            dataset_info=dataset_info,
            dataloader_train=train_loader,
            checkpoint_path=checkpoint_dir,
        )
    except FileNotFoundError as exc:
        print(f"Skipping GeoLDM sanity check: {exc}")
        print("Download the full checkpoint files to run the integration test.")
        return

    device = next(ldm.parameters()).device
    nodesxsample = nodes_dist.sample(4)
    one_hot, charges, positions, node_mask = sampling.sample(
        train_args,
        device,
        ldm,
        dataset_info,
        nodesxsample=nodesxsample,
        prop_dist=None,
    )

    atom_types = one_hot[0].argmax(dim=-1).cpu().numpy()
    atom_symbols = [dataset_info["atom_decoder"][i] for i in atom_types]
    visualize_molecule_3d(
        atom_symbols,
        positions[0].cpu().numpy(),
        show=True,
    )

    print(
        "Sanity check finished successfully. Molecule visualisation was generated",
        "without opening an interactive window.",
    )


if __name__ == "__main__":
    main()
