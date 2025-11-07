# GeoLDM Core Modules

이 저장소는 GeoLDM 원본 레포지토리에서 모델 실행에 필요한 최소 모듈만 분리하여 정리한 버전입니다.
`geoldm` 패키지는 모델 정의, 사전 학습 체크포인트 로딩, 샘플링 및 시각화에 필요한 유틸리티만 포함하도록 가볍게 구성되어 있습니다.

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

아래 예시는 QM9 잠재 확산 모델 체크포인트를 불러와 샘플을 생성하고 시각화하는 전형적인 워크플로우입니다.

```python
from pathlib import Path
import pickle
import torch

from geoldm.configs import get_dataset_info
from geoldm.qm9 import dataset, load_model, sampling, visualize_molecule_3d

# 1. 학습 시 저장된 args.pickle 로드
checkpoint_dir = Path("outputs/geoldm_qm9")
with checkpoint_dir.joinpath("args.pickle").open("rb") as f:
    train_args = pickle.load(f)

# 2. 데이터셋 정보 & 데이터로더 준비
dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h)
dataloaders, _ = dataset.retrieve_dataloaders(train_args)
train_loader = dataloaders["train"]

# 3. 잠재 확산 모델 복원
ldm, nodes_dist, _ = load_model(
    stage="latent_diffusion",
    args=train_args,
    dataset_info=dataset_info,
    dataloader_train=train_loader,
    checkpoint_path=checkpoint_dir,
)

# 4. 샘플링
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

# 5. 첫 번째 분자를 시각화
positions, features = samples[0], samples[1]
one_hot = features["categorical"][0].cpu().numpy()
charges = features["integer"][0].cpu().numpy()

atom_types = one_hot.argmax(axis=-1)
atom_symbols = [dataset_info["atom_decoder"][i] for i in atom_types]
visualize_molecule_3d(atom_symbols, positions[0].cpu().numpy())
```

### 주요 모듈 요약

- `geoldm.load_model`: 체크포인트를 로드하고 지정한 단계(diffusion/autoencoder/latent_diffusion)의 모델을 반환합니다.
- `geoldm.qm9.dataset.retrieve_dataloaders`: QM9/GEOM 데이터셋 로더를 구성합니다.
- `geoldm.qm9.sampling`: 확산 모델에서 샘플을 생성하거나 체인/애니메이션을 만드는 함수 모음입니다.
- `geoldm.qm9.model_module`: 인코딩/디코딩, RDKit 기반 구조 변환, 간단한 시각화 헬퍼를 제공합니다.

필요 시 `geoldm.egnn`, `geoldm.equivariant_diffusion` 하위 모듈을 직접 임포트하여 네트워크를 커스터마이징할 수 있습니다.

## 라이선스

원본 GeoLDM 프로젝트는 MIT 라이선스를 따릅니다. 이 정리본 또한 동일한 라이선스 정책을 적용합니다.
