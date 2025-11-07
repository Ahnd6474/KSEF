"""Utilities for loading GeoLDM models and working with molecular data.

This module provides high level helpers that make it easier to
instantiate the different GeoLDM models and interact with them.
It exposes convenience wrappers for encoding, decoding and sampling
with the diffusion models together with chemistry utilities such as
SMILES/3D interconversion and visualisation helpers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)

from qm9.models import get_autoencoder, get_latent_diffusion, get_model

try:  # pragma: no cover - rdkit is optional and heavy to import during tests
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
except ModuleNotFoundError:  # pragma: no cover - graceful degradation when rdkit is absent
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    rdDetermineBonds = None  # type: ignore


__all__ = [
    "load_model",
    "encode",
    "decode",
    "run_diffusion",
    "visualize_molecule_3d",
    "smiles_to_3d",
    "structure_to_smiles",
]


def _resolve_checkpoint_path(
    checkpoint: Path,
    prefer_ema: bool = True,
) -> Path:
    """Return a concrete checkpoint file from ``checkpoint``.

    ``checkpoint`` can either point to a directory containing the training
    artefacts or to a file.  When a directory is supplied, the helper looks
    for the filenames used by the training scripts (``generative_model.npy``
    and ``generative_model_ema.npy``).  ``prefer_ema`` controls whether the
    exponential-moving-average weights are preferred when they are present.
    """

    if checkpoint.is_file():
        return checkpoint

    candidates: List[str] = [
        "generative_model_ema.npy",
        "generative_model.npy",
    ]
    if not prefer_ema:
        candidates.reverse()

    for name in candidates:
        candidate = checkpoint / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate a checkpoint inside {checkpoint!s}. "
        "Expected one of: " + ", ".join(candidates)
    )


def load_model(
    stage: str,
    args: Any,
    dataset_info: Dict[str, Any],
    dataloader_train: Any,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    prefer_ema: bool = True,
) -> Tuple[torch.nn.Module, Any, Any]:
    """Instantiate and optionally load a GeoLDM model.

    Parameters
    ----------
    stage:
        Which model to construct.  Supported values are ``"diffusion"``
        (first stage diffusion model), ``"autoencoder"`` and
        ``"latent_diffusion"`` (two-stage latent diffusion).
    args:
        Namespace containing the configuration parameters used during
        training.  The functions from :mod:`qm9.models` expect the same
        arguments that are provided by the training scripts.
    dataset_info:
        Metadata describing the dataset, usually obtained via
        :func:`configs.datasets_config.get_dataset_info`.
    dataloader_train:
        Training dataloader.  Only the metadata (e.g. histograms) are
        required, therefore passing the training dataloader is sufficient.
    checkpoint_path:
        Optional path to a checkpoint produced by the training scripts.
        The helper accepts either the direct file path or the directory
        that contains the saved weights.
    device:
        Device where the model should be placed.  When ``None`` the device is
        inferred from ``args.cuda`` if available, otherwise the CPU is used.
    prefer_ema:
        When ``True`` and the directory contains EMA weights, they are loaded
        instead of the raw model parameters.

    Returns
    -------
    tuple
        ``(model, nodes_dist, prop_dist)`` exactly as returned by the
        respective constructor in :mod:`qm9.models`.
    """

    if device is None:
        use_cuda = bool(getattr(args, "cuda", torch.cuda.is_available()))
        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    stage = stage.lower()
    if stage == "diffusion":
        model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloader_train)
    elif stage == "autoencoder":
        model, nodes_dist, prop_dist = get_autoencoder(args, device, dataset_info, dataloader_train)
    elif stage in {"latent_diffusion", "latent-diffusion"}:
        model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloader_train)
    else:
        raise ValueError(
            "Unsupported stage '{stage}'. Expected 'diffusion', 'autoencoder' or 'latent_diffusion'.".format(
                stage=stage
            )
        )

    model.to(device)

    if checkpoint_path is not None:
        checkpoint = _resolve_checkpoint_path(Path(checkpoint_path), prefer_ema=prefer_ema)
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    return model, nodes_dist, prop_dist


def encode(
    model: torch.nn.Module,
    x: torch.Tensor,
    h: Dict[str, torch.Tensor],
    node_mask: Optional[torch.Tensor] = None,
    edge_mask: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """Encode molecular inputs with the provided model.

    The helper transparently supports models that expose an ``encode``
    method (such as :class:`equivariant_diffusion.en_diffusion.EnHierarchicalVAE`)
    as well as the latent diffusion model where the encoder is stored under
    the ``vae`` attribute.
    """

    if hasattr(model, "encode"):
        return model.encode(x, h, node_mask=node_mask, edge_mask=edge_mask, context=context)
    if hasattr(model, "vae") and hasattr(model.vae, "encode"):
        return model.vae.encode(x, h, node_mask=node_mask, edge_mask=edge_mask, context=context)
    raise AttributeError("The supplied model does not expose an encoder.")


def decode(
    model: torch.nn.Module,
    z: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    edge_mask: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Decode latent representations back to atom positions and features."""

    if hasattr(model, "decode"):
        decoded = model.decode(z, node_mask=node_mask, edge_mask=edge_mask, context=context)
        if isinstance(decoded, tuple):
            return decoded  # type: ignore[return-value]
        raise ValueError("Model.decode returned an unexpected object: {!r}".format(type(decoded)))

    if hasattr(model, "vae") and hasattr(model.vae, "decode"):
        return model.vae.decode(z, node_mask=node_mask, edge_mask=edge_mask, context=context)

    raise AttributeError("The supplied model does not expose a decoder.")


@torch.no_grad()
def run_diffusion(
    model: torch.nn.Module,
    n_samples: int,
    n_nodes: int,
    node_mask: torch.Tensor,
    edge_mask: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    fix_noise: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sample molecules with a diffusion model."""

    if not hasattr(model, "sample"):
        raise AttributeError("The supplied model does not implement sampling.")

    samples = model.sample(
        n_samples,
        n_nodes,
        node_mask,
        edge_mask,
        context,
        fix_noise=fix_noise,
    )
    if isinstance(samples, tuple):
        return samples  # type: ignore[return-value]
    raise ValueError("Model.sample returned an unexpected object: {!r}".format(type(samples)))


def visualize_molecule_3d(
    atom_symbols: Sequence[str],
    coordinates: np.ndarray,
    bonds: Optional[Iterable[Tuple[int, int]]] = None,
    *,
    title: Optional[str] = None,
    annotate: bool = False,
    ax: Optional[Axes3D] = None,
    atom_colors: Optional[Dict[str, str]] = None,
    show: bool = True,
) -> Axes3D:
    """Plot a molecule in 3D using :mod:`matplotlib`.

    Parameters
    ----------
    atom_symbols:
        Iterable with the atomic symbols (e.g. ``["C", "O", "H"]``).
    coordinates:
        Array of shape ``(N, 3)`` containing the positions of each atom.
    bonds:
        Optional iterable of pairs ``(i, j)`` describing bonds between the
        atoms.  Bonds are rendered as straight line segments.
    title:
        Optional title for the plot.
    annotate:
        When ``True`` the atom indices are displayed next to each point.
    ax:
        Optional existing 3D axis.  When omitted a new figure is created.
    atom_colors:
        Mapping from atomic symbol to matplotlib colour codes.  Defaults to a
        simple palette if ``None``.
    show:
        Whether ``plt.show()`` should be executed when creating a new figure.
    """

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if len(atom_symbols) != coords.shape[0]:
        raise ValueError("atom_symbols and coordinates must contain the same number of atoms")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = None

    default_colors = {
        "H": "#CCCCCC",
        "C": "#1f77b4",
        "N": "#2ca02c",
        "O": "#d62728",
        "F": "#9467bd",
        "S": "#ff7f0e",
        "Cl": "#17becf",
    }
    palette = default_colors.copy()
    if atom_colors:
        palette.update(atom_colors)

    for idx, (symbol, (x, y, z)) in enumerate(zip(atom_symbols, coords)):
        color = palette.get(symbol, "#7f7f7f")
        ax.scatter(x, y, z, color=color, s=60, depthshade=True)
        if annotate:
            ax.text(x, y, z, f"{symbol}{idx}", fontsize=8, ha="center")

    if bonds is not None:
        for bond in bonds:
            if len(bond) != 2:
                raise ValueError("Bond specification must contain two atom indices")
            i, j = bond
            segment = np.vstack([coords[i], coords[j]])
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#444444", linewidth=2)

    max_range = (coords.max(axis=0) - coords.min(axis=0)).max()
    center = coords.mean(axis=0)
    radius = max_range / 2 or 1.0
    for axis, coord in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        axis(coord - radius, coord + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title)

    if fig is not None and show:
        plt.show()

    return ax


def smiles_to_3d(
    smiles: str,
    num_conformers: int = 1,
    optimize: bool = True,
    random_seed: Optional[int] = 0,
) -> List[Dict[str, Any]]:
    """Convert a SMILES string into 3D coordinates using RDKit.

    Returns a list of dictionaries containing the RDKit molecule, the conformer
    identifier, atomic symbols and the coordinates for each generated conformer.
    """

    if Chem is None or AllChem is None:
        raise ImportError("RDKit is required for SMILES to 3D conversion but is not installed.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES string: {smiles!r}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    if random_seed is not None:
        params.randomSeed = random_seed

    conformer_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params))
    if optimize:
        for conf_id in conformer_ids:
            try:
                AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            except Exception:
                # Optimisation occasionally fails for unusual molecules.
                pass

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    conformers: List[Dict[str, Any]] = []
    for conf_id in conformer_ids:
        conformer = mol.GetConformer(conf_id)
        coords = np.array(conformer.GetPositions(), dtype=float)
        conformers.append(
            {
                "mol": mol,
                "conformer_id": conf_id,
                "atom_symbols": atom_symbols,
                "coordinates": coords,
            }
        )

    return conformers


def structure_to_smiles(
    atom_symbols: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    charge: int = 0,
    sanitize: bool = True,
) -> Tuple[str, Any]:
    """Convert 3D coordinates into a (canonical) SMILES string using RDKit."""

    if Chem is None or rdDetermineBonds is None:
        raise ImportError("RDKit is required for 3D to SMILES conversion but is not installed.")

    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (N, 3)")
    if len(atom_symbols) != coords.shape[0]:
        raise ValueError("atom_symbols and coordinates must contain the same number of atoms")

    # Construct an XYZ block as intermediate representation.
    header = f"{len(atom_symbols)}\nConverted by GeoLDM\n"
    body_lines = [f"{symbol} {x:.6f} {y:.6f} {z:.6f}" for symbol, (x, y, z) in zip(atom_symbols, coords)]
    xyz_block = header + "\n".join(body_lines) + "\n"

    mol = Chem.RWMol(Chem.rdmolfiles.MolFromXYZBlock(xyz_block))
    if mol is None:
        raise ValueError("Could not create an RDKit molecule from the provided coordinates.")

    rdDetermineBonds.DetermineConnectivity(mol, charge=charge)
    rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
    mol = mol.GetMol()

    if sanitize:
        Chem.SanitizeMol(mol)

    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    return smiles, mol
