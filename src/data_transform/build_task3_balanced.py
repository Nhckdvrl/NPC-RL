"""
构建 task3 balanced parquet：stage_0 过采样到与 stage_1 各占50%。
输出 data/verl/task3_balanced_train.parquet / val.parquet
"""
import pandas as pd, numpy as np
from pathlib import Path

OUT = Path("data/verl")

s0_tr = pd.read_parquet(OUT / "stage_0_train.parquet")
s1_tr = pd.read_parquet(OUT / "stage_1_train.parquet")
s0_va = pd.read_parquet(OUT / "stage_0_val.parquet")
s1_va = pd.read_parquet(OUT / "stage_1_val.parquet")

# 过采样 stage_0 到与 stage_1 一样多
rng = np.random.default_rng(42)
n = len(s1_tr)
idx = rng.choice(len(s0_tr), size=n, replace=True)
s0_up = s0_tr.iloc[idx].reset_index(drop=True)

train = pd.concat([s0_up, s1_tr], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
val   = pd.concat([s0_va, s1_va], ignore_index=True).reset_index(drop=True)

train.to_parquet(OUT / "task3_balanced_train.parquet", index=False)
val.to_parquet(OUT / "task3_balanced_val.parquet",   index=False)
print(f"train: {len(train)} ({len(s0_up)} toolcall + {len(s1_tr)} roleplay)")
print(f"val:   {len(val)} ({len(s0_va)} toolcall + {len(s1_va)} roleplay)")
