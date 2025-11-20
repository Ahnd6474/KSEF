# GeoLDM Core Modules

이 저장소는 GeoLDM 원본 레포지토리에서 모델 실행에 필요한 최소 모듈만 분리하여 정리한 버전입니다.
`geoldm` 패키지는 모델 정의, 사전 학습 체크포인트 로딩, 샘플링 및 시각화에 필요한 유틸리티만 포함하도록 가볍게 구성되어 있습니다
.

## 폴더 구조

```
├── geoldm/
│   ├── __init__.py                # GeoLDM 핵심 진입점
│   ├── configs/                   # 데이터셋 메타 정보
│   ├── egnn/                      # EGNN 기반 네트워크 구현
│   ├── equivariant_diffusion/     # (잠재) 확산 모델 구성 요소
│   ├── geom/                      # GEOM-Drugs 데이터 로더 보조 모듈
│   └── qm9/                       # QM9 데이터 및 모델 유틸리티
└── data_plastic.parquet           # 사용자가 추가한 데이터 (예시)
```

## 설치

필요한 의존성은 원본 GeoLDM 프로젝트와 동일합니다. 대표적으로 다음 패키지가 필요합니다.

- Python ≥ 3.9
- PyTorch ≥ 1.12
- NumPy, SciPy
- Matplotlib (시각화)
- RDKit (SMILES/3D 변환 기능을 사용할 경우)

의존성 관리는 환경마다 다르므로, 기존 GeoLDM 환경을 그대로 활용하는 것을 권장합니다.

## 사용 방법

아래 예시는 "SMILES → 3D → 잠재 공간 확산 → 3D 복원 → SMILES"의 전체 흐름을 최소 코드로 보여줍니다.
각 단계가 함수로 분리되어 있어 다른 프로젝트나 노트북으로 쉽게 옮겨 쓸 수 있습니다.

```python
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from geoldm import (
    decode,
    encode,
    load_model,
    run_diffusion,
    smiles_to_3d,
    structure_to_smiles,
    visualize_molecule_3d,
)
from geoldm.configs import get_dataset_info
from geoldm.qm9 import dataset


def load_qm9_latent_diffusion(checkpoint_dir: Path):
    """Load a pretrained latent diffusion model and its dataset metadata."""

    args_path = checkpoint_dir / "args.pickle"
    with args_path.open("rb") as handle:
        args = pickle.load(handle)

    # CPU에서도 바로 실행할 수 있도록 CUDA 플래그를 강제 조정.
    if not torch.cuda.is_available():
        setattr(args, "cuda", False)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    dataloaders, _ = dataset.retrieve_dataloaders(args)
    train_loader = dataloaders["train"]

    model, nodes_dist, _ = load_model(
        stage="latent_diffusion",
        args=args,
        dataset_info=dataset_info,
        dataloader_train=train_loader,
        checkpoint_path=checkpoint_dir,
    )
    device = next(model.parameters()).device
    return model, dataset_info, nodes_dist, device


def conformer_to_tensors(
    conformer: Dict,
    dataset_info: Dict,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convert a single RDKit conformer into GeoLDM-ready tensors."""

    atom_decoder: List[str] = dataset_info["atom_decoder"]
    atom_encoder = {symbol: idx for idx, symbol in enumerate(atom_decoder)}
    atom_indices = torch.tensor(
        [atom_encoder[symbol] for symbol in conformer["atom_symbols"]],
        dtype=torch.long,
        device=device,
    )

    one_hot = F.one_hot(atom_indices, num_classes=len(atom_decoder)).float()
    positions = torch.tensor(conformer["coordinates"], dtype=torch.float32, device=device)

    # Batch 차원을 추가하고 마스크를 구성합니다.
    x = positions.unsqueeze(0)
    h = {
        "categorical": one_hot.unsqueeze(0),
        # 전하 정보가 없으면 0으로 채웁니다.
        "integer": torch.zeros(one_hot.shape[0], 1, device=device).unsqueeze(0),
    }

    node_mask = torch.ones(x.shape[0], x.shape[1], 1, device=device)
    edge_mask = node_mask.squeeze(-1)[..., None] * node_mask.squeeze(-1)[:, None]
    return x, h, node_mask, edge_mask


def main():
    checkpoint_dir = Path("./qm9_latent2")
    model, dataset_info, nodes_dist, device = load_qm9_latent_diffusion(checkpoint_dir)

    # 1) SMILES → 3D 좌표
    conformer = smiles_to_3d("CCO")[0]
    x, h, node_mask, edge_mask = conformer_to_tensors(conformer, dataset_info, device)

    # 2) 3D 구조 → LDM 인코더 잠재벡터
    z_x, z_sigma_x, z_h, z_sigma_h = encode(model, x, h, node_mask=node_mask, edge_mask=edge_mask)

    # 3) 인코딩 벡터를 확산 모델로 보정/샘플링
    n_nodes = x.size(1)
    n_samples = 1
    # GeoLDM의 확산 샘플러는 원래 nodes_dist를 사용하므로 동일한 노드 수로 호출합니다.
    samples = run_diffusion(
        model,
        n_samples=n_samples,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        context=None,
        fix_noise=True,
    )
    sampled_z, _ = samples  # (latent_x, latent_h)

    # 4) 잠재벡터 → 3D 구조 복원
    decoded_positions, decoded_features = decode(
        model,
        sampled_z,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )

    # 5) 3D 구조 → SMILES
    atom_types = decoded_features["categorical"].argmax(dim=-1)[0].cpu().numpy()
    atom_symbols = [dataset_info["atom_decoder"][i] for i in atom_types]
    new_smiles, _ = structure_to_smiles(atom_symbols, decoded_positions[0].cpu().numpy())
    print(f"Generated SMILES: {new_smiles}")

    # 6) 구조 저장/시각화
    visualize_molecule_3d(
        atom_symbols,
        decoded_positions[0].cpu().numpy(),
        out_html="molecule_visualization.html",
        show=False,
    )
    print("Visualization saved to molecule_visualization.html")


if __name__ == "__main__":
    main()
```

### 주요 모듈 요약

- `geoldm.load_model`: 체크포인트를 로드하고 지정한 단계(diffusion/autoencoder/latent_diffusion)의 모델을 반환합니다.
- `geoldm.qm9.dataset.retrieve_dataloaders`: QM9/GEOM 데이터셋 로더를 구성합니다.
- `geoldm.qm9.sampling`: 확산 모델에서 샘플을 생성하거나 체인/애니메이션을 만드는 함수 모음입니다.
- `geoldm.qm9.model_module`: 인코딩/디코딩, RDKit 기반 구조 변환, 간단한 시각화 헬퍼를 제공합니다.

필요 시 `geoldm.egnn`, `geoldm.equivariant_diffusion` 하위 모듈을 직접 임포트하여 네트워크를 커스터마이징할 수 있습니다.

## 라이선스

원본 GeoLDM 프로젝트는 MIT 라이선스를 따릅니다. 이 정리본 또한 동일한 라이선스 정책을 적용합니다.
