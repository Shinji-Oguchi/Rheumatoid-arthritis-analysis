# ============================================
# Rheumatoid Arthritis Analysis with BMF
# ============================================

# ============================================================
# ## BMF Complete Code
# ============================================================

# ============================================================
# ### Import library
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nimfa

# ============================================================
# ### Check versions of the libraries
# ============================================================

import sys
import pkg_resources

print("=" * 60)
print(f"Python Version: {sys.version.split()[0]}")
print("=" * 60)
print("\n主な分析関連ライブラリ：")
print("-" * 60)

# 主な分析関連ライブラリ
main_libraries = [
    'pandas',
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'nimfa'
]

# インストール済みパッケージから検索
installed_dict = {d.key: d.version for d in pkg_resources.working_set}

for lib in main_libraries:
    if lib in installed_dict:
        print(f"{lib}: {installed_dict[lib]}")

print("=" * 60)

# ============================================================
# ## 1. Data processing
# ============================================================

# ============================================================
# ### 1-1. Change names of the columns
# ============================================================

ra_data_all = pd.read_csv("A01.rmdup.agesex.txt", sep="\t")

ra_data_all["A-NO"].nunique()

ra_data_personal = ra_data_all.loc[:,"A-NO":"BOOLEAN"]
ra_data_personal.columns = ['ID', 'date', 'F2_TJC', 'F2_SJC', 'F_PtVAS_mm', 'F_DrVAS_mm', 'F_ESR',
       'F_CRP', 'DAS28-ESR', 'DAS28-CRP', 'CDAI', 'SDAI', 'BOOLEAN']
ra_data_personal_2 = ra_data_all.loc[:,"AGE":"SEX"]
ra_data_personal_3 = pd.concat([ra_data_personal, ra_data_personal_2], axis=1)

ra_data_personal_3.to_csv("RA_personal_20251208.csv")

ra_data_joint = ra_data_all.loc[:,"F_CLICKMAP_0":"F_CLICKMAP_70"]

# 関節番号と対応する関節名を定義
joint_mapping = {
    0: "jaw_r",
    1: "jaw_l",
    2: "neck",
    3: "sternoclavicular_r",
    4: "sternoclavicular_l",
    5: "shoulder_r",
    6: "shoulder_l",
    7: "chest_r",
    8: "chest_l",
    9: "elbow_r",
    10: "elbow_l",
    11: "wrist_r",
    12: "wrist_l",
    13: "hip_r",
    14: "hip_l",
    15: "knee_r",
    16: "knee_l",
    17: "ankle_r",
    18: "ankle_l",
    19: "heel_r",
    20: "heel_l",
    21: "cm_r",
    22: "mcp1_r",
    23: "ip_r",
    24: "mcp5_r",
    25: "mcp4_r",
    26: "mcp3_r",
    27: "mcp2_r",
    28: "pip5_r",
    29: "pip4_r",
    30: "pip3_r",
    31: "pip2_r",
    32: "dip5_r",
    33: "dip4_r",
    34: "dip3_r",
    35: "dip2_r",
    36: "cm_l",
    37: "mcp1_l",
    38: "ip_l",
    39: "mcp2_l",
    40: "mcp3_l",
    41: "mcp4_l",
    42: "mcp5_l",
    43: "pip2_l",
    44: "pip3_l",
    45: "pip4_l",
    46: "pip5_l",
    47: "dip2_l",
    48: "dip3_l",
    49: "dip4_l",
    50: "dip5_l",
    51: "mtp5_r",
    52: "mtp4_r",
    53: "mtp3_r",
    54: "mtp2_r",
    55: "mtp1_r",
    56: "toe_pip5_r",
    57: "toe_pip4_r",
    58: "toe_pip3_r",
    59: "toe_pip2_r",
    60: "toe_pip1_r",
    61: "mtp1_l",
    62: "mtp2_l",
    63: "mtp3_l",
    64: "mtp4_l",
    65: "mtp5_l",
    66: "toe_pip1_l",
    67: "toe_pip2_l",
    68: "toe_pip3_l",
    69: "toe_pip4_l",
    70: "toe_pip5_l"
}

# 対応表をDataFrameで表示
joint_df = pd.DataFrame(list(joint_mapping.items()), columns=["F_CLICKMAP_ID", "Joint_Name"])

# F_CLICKMAP_番号 → Joint_Name へのマッピング辞書を作る
column_rename_map = {
    f"F_CLICKMAP_{i}": name for i, name in joint_mapping.items()
}

# DataFrame に変換を適用
ra_data_joint_renamed = ra_data_joint.rename(columns=column_rename_map)

# ============================================================
# ### 1-2. Tenderness data and Swelling data
# ============================================================

df = ra_data_joint_renamed.copy()

# 圧痛（1,3 → 1、それ以外 → 0）
tender_df = df.copy().apply(lambda col: col.map({1: 1, 3: 1, 0: 0, 2: 0}))

# 腫脹（2,3 → 1、それ以外 → 0）
swollen_df = df.copy().apply(lambda col: col.map({2: 1, 3: 1, 0: 0, 1: 0}))

tender_df.columns = [f"tender_{col}" for col in tender_df.columns]
swollen_df.columns = [f"swollen_{col}" for col in swollen_df.columns]

tender_df.to_csv("tender_joint_matrix_20251208.csv", index=False)
swollen_df.to_csv("swollen_joint_matrix_20251208.csv", index=False)

# ============================================================
# ### 1-3. Exclude joins less than 1%
# ============================================================

tender_df = pd.read_csv("tender_joint_matrix_20251208.csv")
swollen_df = pd.read_csv("swollen_joint_matrix_20251208.csv")

# しきい値（割合）を指定
threshold = 0.01  # 1%

filtered_tender_df = tender_df.loc[:, (tender_df.sum(axis=0) / tender_df.shape[0]) >= threshold]
filtered_swollen_df = swollen_df.loc[:, (swollen_df.sum(axis=0) / swollen_df.shape[0]) >= threshold]

filtered_tender_df.to_csv("filtered_tender_joint_matrix_20251208.csv", index=False)
filtered_swollen_df.to_csv("filtered_swollen_joint_matrix_20251208.csv", index=False)

print(filtered_tender_df.shape)
print(filtered_swollen_df.shape)

# しきい値（割合）を指定
threshold = 0.01  # 1%

eliminated_tender_df = tender_df.loc[:, (tender_df.sum(axis=0) / tender_df.shape[0]) < threshold]
eliminated_swollen_df = swollen_df.loc[:, (swollen_df.sum(axis=0) / swollen_df.shape[0]) < threshold]

# ============================================================
# ## 2.Apply BMF to Tenderness dataset
# ============================================================

# ============================================================
# ### 2-1. Choose the optimal rank
# ============================================================

# ============================================================
# #### mask_ratio = 0.01
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.01, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_tender_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.01
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_tenderness_maskratio0.01_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.05
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.05, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_tender_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.05
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_tenderness_maskratio0.05_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.10
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.1, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_tender_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.1
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_tenderness_maskratio0.10_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.15
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.15, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_tender_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.15
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_tenderness_maskratio0.15_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.20
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.2, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_tender_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.20
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_tenderness_maskratio0.20_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.30
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.3, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_tender_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.30
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_tenderness_maskratio0.30_20251208.png")
plt.show()


# ============================================================
# ### 2-2. Grid search of BMF
# ============================================================

import itertools
from tqdm import tqdm

# RSS計算関数
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 固定マスク生成
def generate_mask_indices(V, mask_ratio=0.1, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

def apply_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# 実行関数
def grid_search_bmf(V, rank, mask_ratio=0.1, random_state=1234):
    lambda_vals = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    seed_options = ['nndsvd']
    param_grid = list(itertools.product(lambda_vals, lambda_vals, seed_options))

    masked_indices = generate_mask_indices(V, mask_ratio=mask_ratio, random_state=random_state)
    V_masked = apply_mask(V, masked_indices)

    results = []
    for lambda_w, lambda_h, seed in tqdm(param_grid, desc=f"Grid Search for rank={rank}"):
        try:
            bmf = nimfa.Bmf(V_masked.values, rank=rank, seed=seed,
                            lambda_w=lambda_w, lambda_h=lambda_h,
                            max_iter=300)
            fit = bmf()
            V_recon = np.dot(fit.basis(), fit.coef())
            rss = evaluate_reconstruction(V.values, V_recon, masked_indices)
            results.append({
                'lambda_w': lambda_w,
                'lambda_h': lambda_h,
                'seed': seed,
                'rss': rss
            })
        except Exception as e:
            print(f"Failed for λ_w={lambda_w}, λ_h={lambda_h}, seed={seed}: {e}")

    result_df = pd.DataFrame(results)
    best_params = result_df.loc[result_df['rss'].idxmin()]
    return result_df, best_params


rss_df_tender, best_tender = grid_search_bmf(filtered_tender_df, rank=2)

best_tender

# ============================================================
# ### 2-3. Tenderness bmf dataset
# ============================================================

bmf = nimfa.Bmf(filtered_tender_df.values, rank=2, seed='nndsvd', lambda_w=1.2, lambda_h=1.2, max_iter=300)
fit = bmf()
W = pd.DataFrame(fit.basis(), index=filtered_tender_df.index)
H = pd.DataFrame(fit.coef(), columns=filtered_tender_df.columns)

H.index = ["Tenderness Basis1", "Tenderness Basis2"]

H.to_csv("Tenderness bases 20251208.csv")

# ============================================================
# ## 3. Apply bmmf to Swelling dataset
# ============================================================

# ============================================================
# ### 3-1. Choose the optimal rank
# ============================================================

# ============================================================
# #### mask_ratio = 0.01
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.01, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_swollen_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.01
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_swelling_maskratio0.01_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.05
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.05, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_swollen_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.05
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_swelling_maskratio0.05_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.10
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.1, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_swollen_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.1
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_swelling_maskratio0.10_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.15
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.15, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_swollen_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.15
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_swelling_maskratio0.15_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.20
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.2, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_swollen_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.20
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_swelling_maskratio0.20_20251208.png")
plt.show()


# ============================================================
# #### mask_ratio = 0.30
# ============================================================

# 固定マスクインデックス生成
def generate_fixed_mask_indices(V, mask_ratio=0.3, random_state=1234):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    np.random.seed(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    return ones_indices[np.random.choice(len(ones_indices), mask_count, replace=False)]

# マスク適用
def apply_fixed_mask(V, mask_indices):
    V_np = V.values.copy()
    for i, j in mask_indices:
        V_np[i, j] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)

# RSS計算
def evaluate_reconstruction(V_original, V_reconstructed, masked_indices):
    rss = 0.0
    for i, j in masked_indices:
        v = V_original[i, j]
        v_hat = V_reconstructed[i, j]
        rss += (v - v_hat) ** 2
    return rss

# 実データ
V = filtered_swollen_df  # あなたのバイナリ行列

# 設定
mask_ratio = 0.30
n_runs = 30
ranks = range(2, 11)
rss_dict = {r: [] for r in ranks}

# 実行
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(V, mask_ratio=mask_ratio, random_state=1000 + run)
    for r in ranks:
        V_masked = apply_fixed_mask(V, fixed_mask_indices)
        bmf = nimfa.Bmf(V_masked.values, rank=r, seed='nndsvd', max_iter=300,
                        lambda_w=1.2, lambda_h=1.2)
        fit = bmf()
        V_recon = np.dot(fit.basis(), fit.coef())
        rss = evaluate_reconstruction(V.values, V_recon, fixed_mask_indices)
        rss_dict[r].append(rss)

# 可視化（箱ひげ図）
df_rss = pd.DataFrame(dict([(f'Rank {r}', pd.Series(rss_dict[r])) for r in ranks]))
plt.figure(figsize=(10, 6))
df_rss.boxplot()
plt.title(f"RSS distribution by Rank (mask_ratio={mask_ratio}, n_runs={n_runs})")
plt.ylabel("Reconstruction Error (RSS)")
plt.xlabel("Rank")
plt.grid(True)
plt.tight_layout()
plt.savefig("optimal_rank_swelling_maskratio0.30_20251208.png")
plt.show()


# ============================================================
# ### 3-2. Swelling bmf dataset
# ============================================================

rss_df_tender, best_tender = grid_search_bmf(filtered_swollen_df, rank=3)

best_tender

bmf = nimfa.Bmf(filtered_swollen_df.values, rank=3, seed='nndsvd', lambda_w=1.2, lambda_h=1.2, max_iter=300)
fit = bmf()
W = pd.DataFrame(fit.basis(), index=filtered_swollen_df.index)
H = pd.DataFrame(fit.coef(), columns=filtered_swollen_df.columns)

H.index = ["Swelling Basis1", "Swelling Basis2", "Swelling Basis3"]

H.to_csv("Swelling bases 20251208.csv")

# ============================================================
# ## 4. Draw Heatmap
# ============================================================

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

df_tender = pd.read_csv("Tenderness bases 20251208.csv", index_col=0)
df_swell = pd.read_csv("Swelling bases 20251208.csv", index_col=0)

df_tender = df_tender.T

df_swell = df_swell.T

ordered_joints = [
    # 右上肢
    'tender_shoulder_r', 'tender_elbow_r', 'tender_wrist_r',
    'tender_cm_r', 'tender_mcp1_r', 'tender_ip_r',
    'tender_mcp2_r', 'tender_mcp3_r', 'tender_mcp4_r', 'tender_mcp5_r',
    'tender_pip2_r', 'tender_pip3_r', 'tender_pip4_r', 'tender_pip5_r',

    # 左上肢
    'tender_shoulder_l', 'tender_elbow_l', 'tender_wrist_l',
    'tender_cm_l', 'tender_mcp1_l', 'tender_ip_l',
    'tender_mcp2_l', 'tender_mcp3_l', 'tender_mcp4_l', 'tender_mcp5_l',
    'tender_pip2_l', 'tender_pip3_l', 'tender_pip4_l', 'tender_pip5_l',

    # 右下肢
    'tender_knee_r', 'tender_ankle_r',
    'tender_mtp1_r', 'tender_mtp2_r', 'tender_mtp3_r', 'tender_mtp4_r',

    # 左下肢
    'tender_knee_l', 'tender_ankle_l',
    'tender_mtp1_l', 'tender_mtp2_l', 'tender_mtp3_l', 'tender_mtp4_l', 'tender_mtp5_l'
]

df_tender = df_tender.loc[ordered_joints]

df_tender.index = df_tender.index.str.replace("^tender_", "", regex=True)

ordered_swollen = [
    # 右上肢
    'swollen_shoulder_r', 'swollen_elbow_r', 'swollen_wrist_r',
    'swollen_cm_r', 'swollen_mcp1_r', 'swollen_ip_r',
    'swollen_mcp2_r', 'swollen_mcp3_r', 'swollen_mcp4_r', 'swollen_mcp5_r',
    'swollen_pip2_r', 'swollen_pip3_r', 'swollen_pip4_r', 'swollen_pip5_r',

    # 左上肢
    'swollen_shoulder_l', 'swollen_elbow_l', 'swollen_wrist_l',
    'swollen_mcp1_l', 'swollen_ip_l',
    'swollen_mcp2_l', 'swollen_mcp3_l', 'swollen_mcp4_l', 'swollen_mcp5_l',
    'swollen_pip2_l', 'swollen_pip3_l', 'swollen_pip4_l', 'swollen_pip5_l',

    # 右下肢
    'swollen_knee_r', 'swollen_ankle_r',
    'swollen_mtp1_r', 'swollen_mtp2_r', 'swollen_mtp3_r', 'swollen_mtp4_r',

    # 左下肢
    'swollen_knee_l', 'swollen_ankle_l',
    'swollen_mtp2_l', 'swollen_mtp3_l'
]


df_swell = df_swell.loc[ordered_swollen]

df_swell.index

df_swell.index = df_swell.index.str.replace("^swollen_", "", regex=True)

# ==== 解剖学順（右上肢・右下肢）====
right_upper = [
    'shoulder_r', 'elbow_r', 'wrist_r',
    'cm_r', 'mcp1_r', 'ip_r',
    'mcp2_r', 'mcp3_r', 'mcp4_r', 'mcp5_r',
    'pip2_r', 'pip3_r', 'pip4_r', 'pip5_r'
]
right_lower = ['knee_r', 'ankle_r', 'mtp1_r', 'mtp2_r', 'mtp3_r', 'mtp4_r', 'mtp5_r']  # 無ければ自動スキップ

# ==== 列ごとの 0–1 スケーリング ====
def col_minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
    X = df.astype(float).copy()
    mn = X.min(axis=0)
    rg = (X.max(axis=0) - mn).replace(0, 1)
    return (X - mn) / rg

tender_scaled = col_minmax_scale(df_tender)
swell_scaled  = col_minmax_scale(df_swell)

# ==== 右上肢 抽出 → 行ユニオン ====
u_rows_t = [r for r in right_upper if r in tender_scaled.index]
u_rows_s = [r for r in right_upper if r in swell_scaled.index]
tender_ru = tender_scaled.loc[u_rows_t]
swell_ru  = swell_scaled.loc[u_rows_s]
u_union   = [r for r in right_upper if (r in u_rows_t) or (r in u_rows_s)]
tender_ru = tender_ru.reindex(u_union)
swell_ru  = swell_ru.reindex(u_union)

# ==== 右下肢 抽出 → 行ユニオン ====
l_rows_t = [r for r in right_lower if r in tender_scaled.index]
l_rows_s = [r for r in right_lower if r in swell_scaled.index]
tender_rl = tender_scaled.loc[l_rows_t]
swell_rl  = swell_scaled.loc[l_rows_s]
l_union   = [r for r in right_lower if (r in l_rows_t) or (r in l_rows_s)]
tender_rl = tender_rl.reindex(l_union)
swell_rl  = swell_rl.reindex(l_union)

# ==== 列：Tend(2)・空白(1)・Swel(3) ====
# 2列/3列前提。列数が足りなければそのまま使える分だけで描画されます
blank_col_u = pd.DataFrame(np.nan, index=u_union, columns=['blank'])
blank_col_l = pd.DataFrame(np.nan, index=l_union, columns=['blank'])
combined_u  = pd.concat([tender_ru.iloc[:, :2], blank_col_u, swell_ru.iloc[:, :3]], axis=1)
combined_l  = pd.concat([tender_rl.iloc[:, :2], blank_col_l, swell_rl.iloc[:, :3]], axis=1)

# ==== 上肢と下肢の間に1行空白 ====
blank_row = pd.DataFrame(np.nan, index=[''], columns=combined_u.columns)

# ==== 縦方向に結合（上肢 → 空行 → 下肢）====
combined = pd.concat([combined_u, blank_row, combined_l], axis=0)

# ==== カラーマップ ====
cyan_cmap    = mcolors.LinearSegmentedColormap.from_list("cyan_cmap", ["white", "cyan"])
magenta_cmap = mcolors.LinearSegmentedColormap.from_list("magenta_cmap", ["white", "magenta"])
orange_cmap  = mcolors.LinearSegmentedColormap.from_list("orange_cmap", ["white", "#FF8000"])
lime_cmap    = mcolors.LinearSegmentedColormap.from_list("lime_cmap", ["white", "#7FFF00"])
yellow_cmap  = mcolors.LinearSegmentedColormap.from_list("yellow_cmap", ["white", "#FFFF00"])
cmaps = [cyan_cmap, magenta_cmap, None, orange_cmap, lime_cmap, yellow_cmap]

# ==== プロット（シンプル版：各値セルを黒枠つき、NaNは完全空欄）====
n_rows, n_cols = combined.shape
fig_h = max(6, 0.45 * n_rows)
fig_w = max(8, 0.6 * n_cols)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_facecolor("white")

for i in range(n_rows):
    for j in range(n_cols):
        v = combined.iat[i, j]
        cmap = cmaps[j] if j < len(cmaps) else None
        if pd.isna(v) or (cmap is None):
            continue  # 空欄（列の空白/行の空白/欠損）は何も描かない
        ax.add_patch(Rectangle((j, i), 1, 1,
                               facecolor=cmap(v), edgecolor='black', linewidth=1.2))

# 軸・ラベル
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.set_aspect('equal'); ax.invert_yaxis()
ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels(list(combined.index), fontsize=18, fontweight="bold")
ax.tick_params(length=0)
ax.set_xlabel(""); ax.set_ylabel("")

# 列ラベル（縦書き・色付き）
ax.xaxis.tick_top()
col_labels = ["Tend-1", "Tend-2", "", "Swel-1", "Swel-2", "Swel-3"][:n_cols]
ax.set_xticklabels(col_labels, rotation=90, fontsize=20, fontweight="bold", va='bottom')
label_colors = ["cyan", "magenta", "black", "#FF8000", "#7FFF00", "#FFFF00"][:n_cols]
for xtick, color in zip(ax.get_xticklabels(), label_colors):
    xtick.set_color(color)

plt.tight_layout()
plt.savefig("RUR+RLR_Tender+Swollen_withRowGap_20251208.pdf", dpi=300)
plt.show()


# ==== 解剖学順（左上肢・左下肢）====
left_upper = [
    'shoulder_l', 'elbow_l', 'wrist_l',
    'cm_l', 'mcp1_l', 'ip_l',
    'mcp2_l', 'mcp3_l', 'mcp4_l', 'mcp5_l',
    'pip2_l', 'pip3_l', 'pip4_l', 'pip5_l'
]
left_lower = ['knee_l', 'ankle_l', 'mtp1_l', 'mtp2_l', 'mtp3_l', 'mtp4_l', 'mtp5_l']  # 無ければ自動スキップ

# ==== 列ごとの 0–1 スケーリング ====
def col_minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
    X = df.astype(float).copy()
    mn = X.min(axis=0)
    rg = (X.max(axis=0) - mn).replace(0, 1)
    return (X - mn) / rg

tender_scaled = col_minmax_scale(df_tender)
swell_scaled  = col_minmax_scale(df_swell)

# ==== 左上肢 抽出 → 行ユニオン ====
u_rows_t = [r for r in left_upper if r in tender_scaled.index]
u_rows_s = [r for r in left_upper if r in swell_scaled.index]
tender_lu = tender_scaled.loc[u_rows_t]
swell_lu  = swell_scaled.loc[u_rows_s]
u_union   = [r for r in left_upper if (r in u_rows_t) or (r in u_rows_s)]
tender_lu = tender_lu.reindex(u_union)
swell_lu  = swell_lu.reindex(u_union)

# ==== 左下肢 抽出 → 行ユニオン ====
l_rows_t = [r for r in left_lower if r in tender_scaled.index]
l_rows_s = [r for r in left_lower if r in swell_scaled.index]
tender_ll = tender_scaled.loc[l_rows_t]
swell_ll  = swell_scaled.loc[l_rows_s]
l_union   = [r for r in left_lower if (r in l_rows_t) or (r in l_rows_s)]
tender_ll = tender_ll.reindex(l_union)
swell_ll  = swell_ll.reindex(l_union)

# ==== 列構成：Tend(2)・空白(1)・Swel(3) ====
blank_col_u = pd.DataFrame(np.nan, index=u_union, columns=['blank'])
blank_col_l = pd.DataFrame(np.nan, index=l_union, columns=['blank'])
combined_u  = pd.concat([tender_lu.iloc[:, :2], blank_col_u, swell_lu.iloc[:, :3]], axis=1)
combined_l  = pd.concat([tender_ll.iloc[:, :2], blank_col_l, swell_ll.iloc[:, :3]], axis=1)

# ==== 上肢と下肢の間に1行空白 ====
blank_row = pd.DataFrame(np.nan, index=[''], columns=combined_u.columns)

# ==== 縦に結合（上肢 → 空行 → 下肢）====
combined = pd.concat([combined_u, blank_row, combined_l], axis=0)

# ==== カラーマップ ====
cyan_cmap    = mcolors.LinearSegmentedColormap.from_list("cyan_cmap", ["white", "cyan"])
magenta_cmap = mcolors.LinearSegmentedColormap.from_list("magenta_cmap", ["white", "magenta"])
orange_cmap  = mcolors.LinearSegmentedColormap.from_list("orange_cmap", ["white", "#FF8000"])
lime_cmap    = mcolors.LinearSegmentedColormap.from_list("lime_cmap",   ["white", "#7FFF00"])
yellow_cmap  = mcolors.LinearSegmentedColormap.from_list("yellow_cmap", ["white", "#FFFF00"])
cmaps = [cyan_cmap, magenta_cmap, None, orange_cmap, lime_cmap, yellow_cmap]

# ==== プロット（値ありセルは黒枠・NAは完全空欄）====
n_rows, n_cols = combined.shape
fig_h = max(6, 0.45 * n_rows)
fig_w = max(8, 0.6 * n_cols)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_facecolor("white")

for i in range(n_rows):
    for j in range(n_cols):
        v = combined.iat[i, j]
        cmap = cmaps[j] if j < len(cmaps) else None
        if pd.isna(v) or (cmap is None):
            continue
        ax.add_patch(Rectangle((j, i), 1, 1,
                               facecolor=cmap(v), edgecolor='black', linewidth=1.2))

# 軸・ラベル
ax.set_xlim(0, n_cols)
ax.set_ylim(0, n_rows)
ax.set_aspect('equal'); ax.invert_yaxis()
ax.set_xticks(np.arange(n_cols) + 0.5)
ax.set_yticks(np.arange(n_rows) + 0.5)
ax.set_yticklabels(list(combined.index), fontsize=18, fontweight="bold")
ax.tick_params(length=0)
ax.set_xlabel(""); ax.set_ylabel("")

# 列ラベル（縦書き・色付き）
ax.xaxis.tick_top()
col_labels  = ["Tend-1", "Tend-2", "", "Swel-1", "Swel-2", "Swel-3"][:n_cols]
ax.set_xticklabels(col_labels, rotation=90, fontsize=20, fontweight="bold", va='bottom')
label_colors = ["cyan", "magenta", "black", "#FF8000", "#7FFF00", "#FFFF00"][:n_cols]
for xtick, color in zip(ax.get_xticklabels(), label_colors):
    xtick.set_color(color)

plt.tight_layout()
plt.savefig("LU+LL_Tender+Swollen_withRowGap_20251208.pdf", dpi=300)
plt.show()
