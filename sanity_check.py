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