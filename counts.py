import pandas as pd
import numpy as np
import ast

path = r"data/plastic.parquet"  # 경로 맞게

df = pd.read_parquet(path)

unique_mols = set()

for raw in df["smiles"]:
    item = raw

    # 1) 문자열이면 literal_eval로 파싱
    if isinstance(item, str):
        try:
            item = ast.literal_eval(item)
        except Exception as e:
            raise ValueError(f"문자열 파싱 실패: {item!r}, 에러: {e}")

    # 2) numpy 배열이면 list로 변환
    if isinstance(item, np.ndarray):
        item = item.tolist()

    # 3) 이제 list/tuple인지 확인
    if not isinstance(item, (list, tuple)):
        raise ValueError(f"예상과 다른 형태의 값(리스트/배열 아님): {repr(item)}")

    if len(item) < 2:
        raise ValueError(f"앞 2개 분자를 찾을 수 없음(길이 < 2): {repr(item)}")

    # [mol1, mol2, (alpha...)] 형태라고 가정하고 앞 두 개만 사용
    mol1, mol2 = item[0], item[1]

    unique_mols.add(mol1)
    unique_mols.add(mol2)

print("고유 분자(모노머) 개수:", len(unique_mols))

# 필요하면 리스트도 보고 싶을 때:
# for m in sorted(unique_mols):
#     print(m)
