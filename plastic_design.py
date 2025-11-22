from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from geoldm import smiles_to_3d

TARGET_COLUMNS = [
    "Tg",
    "Tm",
    "Td",
    "YM",
    "TS_y",
    "TS_b",
    "eps_b",
    "perm_O2",
    "perm_CO2",
    "perm_He",
    "perm_N2",
    "perm_CH4",
    "perm_H2",
]


@dataclass
class ConstraintConfig:
    Tg_req: float
    Td_req: float
    YM_min: float
    TSb_min: float
    epsb_min: float


@dataclass
class LambdaConfig:
    Tg: float = 1.0
    Td: float = 1.0
    YM: float = 1.0
    TS: float = 1.0
    eps: float = 1.0
    z: float = 1e-2


@dataclass
class WeightConfig:
    perm: float = 1.0
    sel_CH4: float = 1.0
    sel_N2: float = 1.0


class MultiTaskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (512, 512, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout
        layers: List[nn.Module] = []
        prev = input_dim
        for width in hidden_sizes:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.SiLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x)


# --- Surrogate helpers ----------------------------------------------------

def _clean_smiles_string(smiles: str) -> str:
    return smiles.replace("[*]", "").strip()


def _pick_base_smiles(smiles_entry: object) -> str:
    def iterate_candidates(seq: Iterable[str]) -> Optional[str]:
        for candidate in seq:
            cleaned = _clean_smiles_string(candidate)
            if cleaned:
                return cleaned
        return None

    if isinstance(smiles_entry, str):
        try:
            maybe_list = ast.literal_eval(smiles_entry)
        except Exception:
            maybe_list = None

        if isinstance(maybe_list, (list, tuple)):
            candidate = iterate_candidates(maybe_list)
            if candidate:
                return candidate

        bracketed = smiles_entry.strip()
        if bracketed.startswith("[") and bracketed.endswith("]"):
            inner = bracketed[1:-1]
            parts = [part.strip() for part in inner.split(",") if part.strip()]
            candidate = iterate_candidates(parts)
            if candidate:
                return candidate

        return _clean_smiles_string(smiles_entry)

    if isinstance(smiles_entry, (list, tuple)):
        candidate = iterate_candidates(smiles_entry)
        if candidate:
            return candidate

    return str(smiles_entry)


def _scaler_to_tensors(scaler: StandardScaler, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "mean": torch.tensor(scaler.mean_, device=device, dtype=torch.float32),
        "scale": torch.tensor(scaler.scale_, device=device, dtype=torch.float32),
    }


def surrogate_forward(latent: torch.Tensor, *, mlp_model: MultiTaskMLP, x_stats: Dict[str, torch.Tensor], y_stats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    latent_scaled = (latent - x_stats["mean"]) / x_stats["scale"]
    preds_scaled = mlp_model(latent_scaled)
    preds = preds_scaled * y_stats["scale"] + y_stats["mean"]
    return {name: preds[:, idx] for idx, name in enumerate(TARGET_COLUMNS)}


def surrogate_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    mlp_model: MultiTaskMLP,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
    weight_config: WeightConfig,
    constraint_config: ConstraintConfig,
    lambda_config: LambdaConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = z1.device
    latent = torch.cat([z1, z2], dim=1)
    props = surrogate_forward(latent, mlp_model=mlp_model, x_stats=x_stats, y_stats=y_stats)

    logP_CO2 = torch.log(torch.clamp(props["perm_CO2"], min=1e-9))
    logP_CH4 = torch.log(torch.clamp(props["perm_CH4"], min=1e-9))
    logP_N2 = torch.log(torch.clamp(props["perm_N2"], min=1e-9))

    J_sep = (
        weight_config.perm * logP_CO2
        + weight_config.sel_CH4 * (logP_CO2 - logP_CH4)
        + weight_config.sel_N2 * (logP_CO2 - logP_N2)
    )

    Tg_req = torch.tensor(constraint_config.Tg_req, device=device)
    Td_req = torch.tensor(constraint_config.Td_req, device=device)
    YM_min = torch.tensor(constraint_config.YM_min, device=device)
    TSb_min = torch.tensor(constraint_config.TSb_min, device=device)
    epsb_min = torch.tensor(constraint_config.epsb_min, device=device)

    pen_Tg = F.relu(Tg_req - props["Tg"]) ** 2
    pen_Td = F.relu(Td_req - props["Td"]) ** 2
    pen_YM = F.relu(YM_min - props["YM"]) ** 2
    pen_TS = F.relu(TSb_min - props["TS_b"]) ** 2
    pen_eps = F.relu(epsb_min - props["eps_b"]) ** 2
    pen_z = (z1 ** 2).sum(dim=1) + (z2 ** 2).sum(dim=1)

    J = (
        J_sep
        - lambda_config.Tg * pen_Tg
        - lambda_config.Td * pen_Td
        - lambda_config.YM * pen_YM
        - lambda_config.TS * pen_TS
        - lambda_config.eps * pen_eps
        - lambda_config.z * pen_z
    )

    loss = -J.mean()
    metrics = {
        "loss": loss.detach(),
        "J": J.mean().detach(),
        "J_sep": J_sep.mean().detach(),
        "pen_Tg": pen_Tg.mean().detach(),
        "pen_Td": pen_Td.mean().detach(),
        "pen_YM": pen_YM.mean().detach(),
        "pen_TS": pen_TS.mean().detach(),
        "pen_eps": pen_eps.mean().detach(),
        "pen_z": pen_z.mean().detach(),
    }
    return loss, metrics


def optimize_latent_with_adam(
    *,
    mlp_model: MultiTaskMLP,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
    weight_config: WeightConfig,
    constraint_config: ConstraintConfig,
    lambda_config: LambdaConfig,
    steps: int = 300,
    lr: float = 5e-3,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, float]]]:
    torch.manual_seed(seed)
    device = next(mlp_model.parameters()).device
    mlp_model.eval()

    latent_dim = mlp_model.network[0].in_features
    split = latent_dim // 2
    z1 = torch.randn(1, split, device=device, requires_grad=True)
    z2 = torch.randn(1, latent_dim - split, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([z1, z2], lr=lr)
    history: List[Dict[str, float]] = []
    best = {"loss": float("inf"), "state": None, "metrics": None, "step": -1}

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        loss, metrics = surrogate_loss(
            z1,
            z2,
            mlp_model=mlp_model,
            x_stats=x_stats,
            y_stats=y_stats,
            weight_config=weight_config,
            constraint_config=constraint_config,
            lambda_config=lambda_config,
        )
        loss.backward()
        optimizer.step()

        metrics = {k: float(v.detach().item()) for k, v in metrics.items()}
        metrics["step"] = step
        history.append(metrics)

        if loss.item() < best["loss"]:
            best = {
                "loss": float(loss.item()),
                "state": (z1.detach().clone(), z2.detach().clone()),
                "metrics": metrics,
                "step": step,
            }

    best_z1, best_z2 = best["state"]  # type: ignore[misc]
    best_latent = torch.cat([best_z1, best_z2], dim=1)
    return best_latent, best["metrics"], history


# --- IO helpers -----------------------------------------------------------

def load_mlp_checkpoint(checkpoint_path: Path, device: torch.device) -> Tuple[MultiTaskMLP, StandardScaler, StandardScaler, Dict]:
    torch.serialization.add_safe_globals([StandardScaler])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MultiTaskMLP(
        checkpoint["input_dim"],
        checkpoint["output_dim"],
        hidden_sizes=checkpoint.get("hidden_sizes", (512, 512, 256)),
        dropout=checkpoint.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint["x_scaler"], checkpoint["y_scaler"], checkpoint


def build_constraint_from_frame(frame: pd.DataFrame) -> ConstraintConfig:
    quantiles = frame[TARGET_COLUMNS].quantile(0.75)
    return ConstraintConfig(
        Tg_req=float(quantiles.get("Tg", 0.0)),
        Td_req=float(quantiles.get("Td", 0.0)),
        YM_min=float(quantiles.get("YM", 0.0)),
        TSb_min=float(quantiles.get("TS_b", 0.0)),
        epsb_min=float(quantiles.get("eps_b", 0.0)),
    )


def predict_properties_from_latent(
    latent: torch.Tensor,
    *,
    mlp_model: MultiTaskMLP,
    x_stats: Dict[str, torch.Tensor],
    y_stats: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    mlp_model.eval()
    with torch.no_grad():
        props = surrogate_forward(latent, mlp_model=mlp_model, x_stats=x_stats, y_stats=y_stats)
    return {key: float(value.squeeze(0).item()) for key, value in props.items()}


def find_nearest_smiles(
    latent: torch.Tensor,
    embeddings: torch.Tensor,
    indices: Sequence[int],
    plastic_df: pd.DataFrame,
) -> Tuple[str, pd.Series, float]:
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    distances = torch.cdist(latent.float(), embeddings.float())
    idx = int(torch.argmin(distances).item())
    row_idx = indices[idx]
    nearest_row = plastic_df.loc[row_idx]
    smiles = _pick_base_smiles(nearest_row["smiles"])
    return smiles, nearest_row, float(distances[0, idx].item())


def conformer_to_xyz(conformer: Dict[str, object], path: Path) -> Path:
    atoms: Sequence[str] = conformer["atom_symbols"]  # type: ignore[index]
    coords: Sequence[Sequence[float]] = conformer["coordinates"]  # type: ignore[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write("Generated from Adam-optimized latent via RDKit 3D conformer\n")
        for atom, (x, y, z) in zip(atoms, coords):
            handle.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")
    return path


# --- CLI -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Adam optimization and decode to a 3D structure.")
    parser.add_argument("--surrogate", type=Path, default=Path("models/plastic_mlp_best.pt"), help="Path to the trained multitask MLP checkpoint.")
    parser.add_argument("--plastic-df", type=Path, default=Path("data/plastic.parquet"), help="Parquet file containing the plastics dataset.")
    parser.add_argument("--latent-cache", type=Path, default=Path("data/plastic_latents.pt"), help="Cached latent embeddings built from the plastics dataset.")
    parser.add_argument("--steps", type=int, default=200, help="Number of Adam optimization steps.")
    parser.add_argument("--lr", type=float, default=5e-3, help="Adam learning rate for latent optimization.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/adam_decode"), help="Directory to store decoded structures and result tables.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for latent initialization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlp_model, x_scaler, y_scaler, _checkpoint = load_mlp_checkpoint(args.surrogate, device)
    x_stats = _scaler_to_tensors(x_scaler, device)
    y_stats = _scaler_to_tensors(y_scaler, device)

    plastic_df = pd.read_parquet(args.plastic_df)
    constraint_config = build_constraint_from_frame(plastic_df)

    best_latent, best_metrics, history = optimize_latent_with_adam(
        mlp_model=mlp_model,
        x_stats=x_stats,
        y_stats=y_stats,
        weight_config=WeightConfig(),
        constraint_config=constraint_config,
        lambda_config=LambdaConfig(),
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )

    prop_predictions = predict_properties_from_latent(
        best_latent,
        mlp_model=mlp_model,
        x_stats=x_stats,
        y_stats=y_stats,
    )

    latent_cache = torch.load(args.latent_cache, map_location=device)
    embeddings: torch.Tensor = latent_cache["embeddings"].to(device)
    indices = latent_cache["indices"]

    smiles, source_row, latent_distance = find_nearest_smiles(best_latent, embeddings, indices, plastic_df)
    conformers = smiles_to_3d(smiles)
    if not conformers:
        raise RuntimeError(f"Failed to generate 3D conformer for SMILES: {smiles}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = conformer_to_xyz(conformers[0], output_dir / "optimized_structure.xyz")

    history_df = pd.DataFrame(history)
    history_path = output_dir / "adam_history.parquet"
    history_df.to_parquet(history_path, index=False)

    summary_row = {
        "smiles": smiles,
        "latent_distance": latent_distance,
        **{f"latent_{i}": float(v) for i, v in enumerate(best_latent.squeeze(0).tolist())},
        **{f"metric_{k}": v for k, v in best_metrics.items() if k != "step"},
        **prop_predictions,
        "source_index": int(source_row.name),
        "xyz_path": xyz_path.as_posix(),
    }
    summary_df = pd.DataFrame([summary_row])
    summary_path = output_dir / "adam_best.parquet"
    summary_df.to_parquet(summary_path, index=False)

    print("\n✅ Adam optimization complete")
    print(f"   Best objective J: {best_metrics['J']:.4f} at step {best_metrics['step']}")
    print(f"   Nearest SMILES: {smiles}")
    print(f"   Saved conformer to: {xyz_path}")
    print(f"   History saved to: {history_path}")
    print(f"   Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
