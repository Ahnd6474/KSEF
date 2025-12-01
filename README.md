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

아래 코드는 `geoldm.qm9.model_module`에 준비된 헬퍼만 이용해 "SMILES → 3D → 잠재 확산 → 3D 복원"을 20줄 내외로 실행하는 예시입니다.

```python
from pathlib import Path

import torch
import torch.nn.functional as F

from geoldm import decode, encode, run_diffusion, smiles_to_3d, visualize_molecule_3d
from geoldm.qm9 import model_module as qm9

# 1) 학습된 잠재 확산 모델 불러오기 (자동으로 CPU/GPU 선택)
model, dataset_info, nodes_dist, device = qm9.load_qm9_latent_diffusion(Path("./qm9_latent2"))

# 2) SMILES를 3D 텐서로 변환
conf = smiles_to_3d("CCO")[0]
atom_decoder = dataset_info["atom_decoder"]
atom_indices = torch.tensor([atom_decoder.index(a) for a in conf["atom_symbols"]], device=device)
x = torch.tensor(conf["coordinates"], device=device, dtype=torch.float32).unsqueeze(0)
h = {
    "categorical": F.one_hot(atom_indices, num_classes=len(atom_decoder)).float().unsqueeze(0),
    "integer": torch.zeros(1, atom_indices.numel(), 1, device=device),
}
node_mask = torch.ones(1, atom_indices.numel(), 1, device=device)
edge_mask = node_mask.squeeze(-1)[..., None] * node_mask.squeeze(-1)[:, None]

# 3) 잠재 확산 샘플링 후 디코딩
sampled_z, _ = run_diffusion(model, 1, atom_indices.numel(), node_mask, edge_mask, context=None, fix_noise=True)
positions, features = decode(model, sampled_z, node_mask=node_mask, edge_mask=edge_mask)
symbols = [atom_decoder[i] for i in features["categorical"].argmax(dim=-1)[0].tolist()]
visualize_molecule_3d(symbols, positions[0].cpu().numpy(), out_html="molecule_visualization.html", show=False)
```

### 주요 모듈 요약

- `geoldm.load_model`: 체크포인트를 로드하고 지정한 단계(diffusion/autoencoder/latent_diffusion)의 모델을 반환합니다.
- `geoldm.qm9.dataset.retrieve_dataloaders`: QM9/GEOM 데이터셋 로더를 구성합니다.
- `geoldm.qm9.sampling`: 확산 모델에서 샘플을 생성하거나 체인/애니메이션을 만드는 함수 모음입니다.
- `geoldm.qm9.model_module`: 인코딩/디코딩, RDKit 기반 구조 변환, 간단한 시각화 헬퍼를 제공합니다.

필요 시 `geoldm.egnn`, `geoldm.equivariant_diffusion` 하위 모듈을 직접 임포트하여 네트워크를 커스터마이징할 수 있습니다.

### 플라스틱 물성 예측 노트북(`plastic.ipynb`)

- `data/plastic.parquet`의 단량체 SMILES를 GeoLDM으로 인코딩해 `latents.pt`에 캐싱한 뒤, `MixtureMLP`로 14개 물성을 동시 학습합니다.
- `mixture_mlp.pt`에는 가중치뿐 아니라 입력/출력 차원(`input_dim`, `output_dim`)과 스케일 정보(`target_means`, `target_stds`)가 함께 저장됩니다.
- PyTorch 2.6 이상에서는 로딩 시 `weights_only=False` 혹은 `torch.serialization.add_safe_globals([...])`를 함께 지정해야 합니다.
- 노트북에 정의된 `MixtureMLP` 클래스를 그대로 가져와 아래처럼 빠르게 예측을 재현할 수 있습니다.

```python
import numpy as np
import torch
from torch.serialization import add_safe_globals
from plastic import MixtureMLP  # 노트북에서 정의한 MLP를 동일하게 사용

add_safe_globals([np._core.multiarray._reconstruct])
ckpt = torch.load("mixture_mlp.pt", map_location="cpu", weights_only=False)
model = MixtureMLP(input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"], hidden_sizes=(512, 256))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

latent = torch.load("latents.pt")
zA, zB = latent["[*]CCCCCC(=O)N[*]"], latent["[*]CCOC(=O)c1ccc(C(=O)O[*])cc1"]
alpha = torch.tensor([0.5])
features = torch.cat([zA, zB, alpha * zA + (1 - alpha) * zB, torch.abs(zA - zB), alpha])
pred_scaled = model(features.unsqueeze(0))
pred = pred_scaled * ckpt["target_stds"] + ckpt["target_means"]
```

### Adam 최적화 + 3D 디코딩 스크립트(`plastic_design.py`)

`plastic_design.py` 스크립트는 플라스틱 물성 surrogate(MLP)를 이용해 잠재벡터를 Adam으로 탐색하고, 결과를 DataFrame으로 저장한 뒤 가장 가까운 실데이터 SMILES를 3D 구조(XYZ)로 저장합니다.

```bash
python plastic_design.py \
  --surrogate models/plastic_mlp_best.pt \
  --plastic-df data/plastic.parquet \
  --latent-cache data/plastic_latents.pt \
  --steps 200 --lr 5e-3 --output-dir runs/adam_decode
```

- `runs/adam_decode/adam_history.parquet`: 최적화 과정의 지표 기록(step, J, penalty 등)
- `runs/adam_decode/adam_best.parquet`: 최적화된 잠재벡터, 예측 물성, 가장 가까운 원본 SMILES 및 XYZ 경로
- `runs/adam_decode/optimized_structure.xyz`: RDKit 기반 3D 좌표 파일(필요 시 다른 뷰어로 시각화 가능)

## 라이선스

원본 GeoLDM 프로젝트는 MIT 라이선스를 따릅니다. 이 정리본 또한 동일한 라이선스 정책을 적용합니다.
