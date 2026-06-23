# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorly as tl
from tensorly.decomposition import non_negative_parafac  

tl.set_backend("numpy")

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

# 　Optimal Rank Selection

filtered_tender_df = pd.read_csv("filtered_tender_joint_matrix_20251208.csv")
filtered_swollen_df = pd.read_csv("filtered_swollen_joint_matrix_20251208.csv")

## Tenderness

### mask_ratio = 0.01

# =========================
# Generate fixed mask indices (select mask_ratio from positions with value 1)
# =========================
def generate_fixed_mask_indices(V: pd.DataFrame, mask_ratio=0.01, random_state=42):
    V_np = V.values.copy()
    ones_indices = np.argwhere(V_np == 1)
    rng = np.random.default_rng(random_state)
    mask_count = int(len(ones_indices) * mask_ratio)
    chosen = rng.choice(len(ones_indices), mask_count, replace=False)
    return ones_indices[chosen]


# =========================
# Apply mask (set the selected 1s to 0)
# =========================
def apply_fixed_mask(V: pd.DataFrame, mask_indices):
    V_np = V.values.copy()
    V_np[mask_indices[:, 0], mask_indices[:, 1]] = 0
    return pd.DataFrame(V_np, index=V.index, columns=V.columns)


# =========================
# MSE calculation (only for masked positions)
# =========================
def evaluate_reconstruction(V_original_np, V_reconstructed_np, masked_indices):
    i = masked_indices[:, 0]
    j = masked_indices[:, 1]
    diff = V_original_np[i, j] - V_reconstructed_np[i, j]
    return float(np.mean(diff ** 2))


# =========================
# NMF equivalent using TensorLy (non-negative CP) -> reconstruction
# =========================
def fit_reconstruct_tensorly_nmf(
    V_masked_np,
    rank,
    n_iter=300,
    random_state=42,
    init="random"
):
    """
    Applying non_negative_parafac to a 2D tensor (matrix) is equivalent to NMF.
    Use cp_to_tensor for reconstruction.
    """
    X = tl.tensor(V_masked_np.astype(float))

    # Reproducibility (absorb environmental differences)
    np.random.seed(random_state)

    cp = non_negative_parafac(
        X,
        rank=rank,
        init=init,
        n_iter_max=n_iter,
        random_state=random_state,
        verbose=False
    )

    X_hat = tl.cp_to_tensor(cp)
    return tl.to_numpy(X_hat)


# =========================
# Data
# =========================
V = filtered_tender_df  
V_np = V.values         


# =========================
# Settings
# =========================
mask_ratio = 0.01
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}


# =========================
# Run
# =========================
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    # Use the same mask for each run
    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)


# =========================
# Box plot visualization
# =========================
df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.01_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.05

# =========================
# Data
# =========================
V = filtered_tender_df  
V_np = V.values         


# =========================
# Settings
# =========================
mask_ratio = 0.05
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}


# =========================
# Run
# =========================
for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)


# =========================
# Box plot visualization
# =========================
df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.05_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.10

V = filtered_tender_df  
V_np = V.values       

mask_ratio = 0.10
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.10_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.15

V = filtered_tender_df 
V_np = V.values        

n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.15_20260201.png",
    dpi=300
)
plt.show()

### mask_ratio = 0.20

V = filtered_tender_df  
V_np = V.values        

mask_ratio = 0.20
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.20_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.25

V = filtered_tender_df  
V_np = V.values        
mask_ratio = 0.25
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.25_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.30

V = filtered_tender_df  
V_np = V.values      

mask_ratio = 0.30
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_tenderness_tensorly_MSE_maskratio0.30_20260201.png",
    dpi=300
)
plt.show()

## Swelling

### mask_ratio = 0.01

V = filtered_swollen_df 
V_np = V.values       

mask_ratio = 0.01
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.01_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.05

V = filtered_swollen_df  
V_np = V.values       

mask_ratio = 0.05
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.05_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.10

V = filtered_swollen_df 
V_np = V.values      

mask_ratio = 0.10
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.10_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.15

V = filtered_swollen_df 
V_np = V.values        

mask_ratio = 0.15
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.15_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.20

V = filtered_swollen_df  
V_np = V.values       

mask_ratio = 0.20
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.20_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.25

V = filtered_swollen_df  
V_np = V.values      

mask_ratio = 0.25
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.25_20260201.png",
    dpi=300
)
plt.show()


### mask_ratio = 0.30

V = filtered_swollen_df  
V_np = V.values       

mask_ratio = 0.30
n_runs = 30
ranks = range(2, 11)

mse_dict = {r: [] for r in ranks}

for run in range(n_runs):
    fixed_mask_indices = generate_fixed_mask_indices(
        V,
        mask_ratio=mask_ratio,
        random_state=1000 + run
    )

    V_masked = apply_fixed_mask(V, fixed_mask_indices)
    V_masked_np = V_masked.values

    for r in ranks:
        V_recon = fit_reconstruct_tensorly_nmf(
            V_masked_np,
            rank=r,
            n_iter=300,
            random_state=2000 + run * 100 + r,
            init="random"
        )

        mse = evaluate_reconstruction(
            V_np,
            V_recon,
            fixed_mask_indices
        )

        mse_dict[r].append(mse)

df_mse = pd.DataFrame(
    {f'Rank {r}': pd.Series(mse_dict[r]) for r in ranks}
)

plt.figure(figsize=(10, 6))
df_mse.boxplot()
plt.title(f"Mask_ratio={mask_ratio}", fontsize=16, fontweight='bold')
plt.ylabel("Mean Squared Error (masked ones)", fontsize=14, fontweight='bold')
plt.xlabel("Rank", fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "optimal_rank_swollen_tensorly_MSE_maskratio0.30_20260201.png",
    dpi=300
)
plt.show()


## Extracting Bases

### Tenderness
# =========================
# data
# =========================
V = filtered_tender_df  # (patients x joints) 

# =========================
# Setteings
# =========================
rank = 4
n_iter = 1000          
random_state = 1234
init = "random"      
tol = 1e-6      

# =========================
# Faactorization
# =========================
X = tl.tensor(V.values.astype(float))

np.random.seed(random_state)

cp = non_negative_parafac(
    X,
    rank=rank,
    init=init,
    n_iter_max=n_iter,
    random_state=random_state,
    verbose=False
)

patients_factor, joints_factor = cp.factors  # shapes: (n_patients, rank), (n_joints, rank)

# =========================
# Create DataFrame
# =========================
basis_cols = [f"Tenderness Basis{i}" for i in range(1, rank + 1)]

df_patient_basis = pd.DataFrame(
    patients_factor,
    index=V.index,       
    columns=basis_cols
)

df_joint_basis = pd.DataFrame(
    joints_factor,
    index=V.columns,     
    columns=basis_cols
)

# =========================
# Reconstruction 
# =========================
X_recon = tl.to_numpy(tl.cp_to_tensor(cp))
df_recon = pd.DataFrame(X_recon, index=V.index, columns=V.columns)

df_patient_basis.to_csv("tensorly_rank4_tenderness_patient_basis_20260201.csv")
df_joint_basis.to_csv("tensorly_rank4_tenderness_joint_basis_20260201.csv")

## Swelling

# =========================
# Data
# =========================
V = filtered_swollen_df   # (patients × joints) 

# =========================
# Settings
# =========================
rank = 3
n_iter = 1000
random_state = 1234
init = "random"     

# =========================
# Factorization
# =========================
X = tl.tensor(V.values.astype(float))

np.random.seed(random_state)

cp = non_negative_parafac(
    X,
    rank=rank,
    init=init,
    n_iter_max=n_iter,
    random_state=random_state,
    verbose=False
)

patients_factor, joints_factor = cp.factors
# shapes:
# patients_factor: (n_patients, 3)
# joints_factor:   (n_joints, 3)

# =========================
# Create DataFrame
# =========================
basis_cols = [f"Swelling Basis{i}" for i in range(1, rank + 1)]

df_patient_basis_swollen = pd.DataFrame(
    patients_factor,
    index=V.index,        
    columns=basis_cols
)

df_joint_basis_swollen = pd.DataFrame(
    joints_factor,
    index=V.columns,     
    columns=basis_cols
)

# =========================
# Reconstruction
# =========================
X_recon = tl.to_numpy(tl.cp_to_tensor(cp))
df_recon_swollen = pd.DataFrame(
    X_recon,
    index=V.index,
    columns=V.columns
)

print("Patient basis shape:", df_patient_basis_swollen.shape)
print("Joint basis shape:", df_joint_basis_swollen.shape)

display(df_patient_basis_swollen.head())
display(df_joint_basis_swollen.head())

df_patient_basis_swollen.to_csv("tensorly_rank3_swelling_patient_basis_20260201.csv")
df_joint_basis_swollen.to_csv("tensorly_rank4_swollen_joint_basis_20260201.csv")


## Combine Tenderness and Swelling Bases

a_joint_basis = pd.read_csv("RA_Joint_Bases_2025618.csv")
ra_joint_basis = ra_joint_basis[["ID", "date", "AGE", "SEX"]]
ra_joint_basis = pd.concat([ra_joint_basis, df_patient_basis], axis=1)
ra_joint_basis = pd.concat([ra_joint_basis, df_patient_basis_swollen], axis=1)
a_joint_basis.to_csv("RA_Patient_Joint_Bases_tensorly_2026201.csv")

ra_joint_basis_sorted = (
    ra_joint_basis
    .assign(
        date=pd.to_datetime(
            ra_joint_basis["date"],
            format="mixed"   
        ).dt.date        
    )
    .sort_values(["ID", "date"])
    .reset_index(drop=True)
)

ra_joint_basis_sorted.to_csv("時系列_RA_Patient_Joint_Bases_tensorly_2026201.csv")

data_ra = pd.read_csv("df_oguchi_20250626.csv")

ra_joint_basis_sorted = ra_joint_basis_sorted.rename(columns={"ID": "Pt"})

ra_joint_basis_sorted["date"] = pd.to_datetime(
    ra_joint_basis_sorted["date"], format="mixed"
).dt.date

data_ra["date"] = pd.to_datetime(
    data_ra["date"], format="mixed"
).dt.date

cols_drug = data_ra.loc[:, "CTLA4_routine_flag":"TNFa_dose"].columns

data_ra_sub = data_ra[
    ["Pt", "date", "SEX", *cols_drug]
].copy()

basis_cols = ra_joint_basis_sorted.loc[
    :, "Tenderness Basis1":"Swelling Basis3"
].columns

ra_basis_sub = ra_joint_basis_sorted[
    ["Pt", "date", *basis_cols]
].copy()

df_merged = ra_basis_sub.merge(
    data_ra_sub,
    on=["Pt", "date"],
    how="left"
)

df_merged.isna().mean().sort_values(ascending=False)

df_merged.to_csv("df_ra_oguchi_20260202.csv")


## Draw Heatmap

# =========================
# Joint Names
# =========================
right_upper = [
    'shoulder', 'elbow', 'wrist',
    'cm', 'mcp1', 'ip',
    'mcp2', 'mcp3', 'mcp4', 'mcp5',
    'pip2', 'pip3', 'pip4', 'pip5'
]
right_lower = ['knee', 'ankle', 'mtp1', 'mtp2', 'mtp3', 'mtp4', 'mtp5']


# =========================
# Column-wise Min-Max Scaling
# =========================
def col_minmax_scale(df: pd.DataFrame) -> pd.DataFrame:
    X = df.astype(float).copy()
    mn = X.min(axis=0, skipna=True)
    rg = (X.max(axis=0, skipna=True) - mn).replace(0, 1)
    return (X - mn) / rg


# =========================
# Colormap 
# =========================
cyan_cmap    = mcolors.LinearSegmentedColormap.from_list("cyan_cmap", ["white", "cyan"])
magenta_cmap = mcolors.LinearSegmentedColormap.from_list("magenta_cmap", ["white", "magenta"])
blue_cmap    = mcolors.LinearSegmentedColormap.from_list("blue_cmap", ["white", "blue"])
red_cmap     = mcolors.LinearSegmentedColormap.from_list("red_cmap", ["white", "red"])

orange_cmap  = mcolors.LinearSegmentedColormap.from_list("orange_cmap", ["white", "#FF8000"])
lime_cmap    = mcolors.LinearSegmentedColormap.from_list("lime_cmap", ["white", "#7FFF00"])
gold_cmap    = mcolors.LinearSegmentedColormap.from_list("gold_cmap", ["white", "gold"])


# =========================
# Normalize Joint Names (remove "tender_" or "swollen_")
# =========================
def normalize_joint_name(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"^(tender_|swollen_)", "", regex=True)
    return s


# =========================
# Split R/L columns and rename (e.g., "shoulder_r" → "Tend–1_R", "shoulder_l" → "Tend–1_L")
# =========================
def split_rl_to_columns(
    df: pd.DataFrame,
    prefix: str,      # "Tend" or "Swel"
    n_basis: int,     # Tend=4, Swel=3
) -> pd.DataFrame:

    d0 = df.copy()
    d0.index = normalize_joint_name(pd.Series(d0.index)).values

    d0.columns = d0.columns.astype(str).str.strip()
    d0.columns = d0.columns.str.replace("–", "-", regex=False).str.replace("—", "-", regex=False).str.replace("−", "-", regex=False)

    d0 = d0.iloc[:, :n_basis].copy()

    # right
    d_r = d0[d0.index.astype(str).str.endswith("_r")].copy()
    d_r.index = d_r.index.astype(str).str.replace("_r$", "", regex=True)
    d_r.columns = [f"{prefix}–{i}_R" for i in range(1, d_r.shape[1] + 1)]

    # left
    d_l = d0[d0.index.astype(str).str.endswith("_l")].copy()
    d_l.index = d_l.index.astype(str).str.replace("_l$", "", regex=True)
    d_l.columns = [f"{prefix}–{i}_L" for i in range(1, d_l.shape[1] + 1)]

    out = pd.concat([d_r, d_l], axis=1)
    out.index.name = "joint"
    return out


# =========================
# Plot Joint Basis Heatmap
# =========================
def plot_joint_basis_heatmap(
    df_joint_basis_tender: pd.DataFrame,   # 圧痛（index *_r/_l, basis列）
    df_joint_basis_swollen: pd.DataFrame,  # 腫脹（index *_r/_l, basis列）
    savepath: str | None = None,
    fontsize: int = 24,
    nan_grey: str = "lightgrey",
):
    df_tend = split_rl_to_columns(df_joint_basis_tender, prefix="Tend", n_basis=4)
    df_swel = split_rl_to_columns(df_joint_basis_swollen, prefix="Swel", n_basis=3)

    df_joint = pd.concat([df_tend, df_swel], axis=1, join="outer")

    cols_tend = []
    for k in range(1, 5):
        for side in ["_R", "_L"]:
            c = f"Tend–{k}{side}"
            if c in df_joint.columns:
                cols_tend.append(c)

    cols_swel = []
    for k in range(1, 4):
        for side in ["_R", "_L"]:
            c = f"Swel–{k}{side}"
            if c in df_joint.columns:
                cols_swel.append(c)

    blank_col = pd.DataFrame(np.nan, index=df_joint.index, columns=["__blank__"])
    df_joint2 = pd.concat([df_joint[cols_tend], blank_col, df_joint[cols_swel]], axis=1)

    # ---- 4) 行順：upper → 空行 → lower ----
    u_rows = [r for r in right_upper if r in df_joint2.index]
    l_rows = [r for r in right_lower if r in df_joint2.index]
    if (len(u_rows) + len(l_rows)) == 0:
        raise ValueError(f"関節名が想定と一致しない。index例: {df_joint2.index[:20].tolist()}")

    combined = pd.concat(
        [
            df_joint2.loc[u_rows],
            pd.DataFrame(np.nan, index=[""], columns=df_joint2.columns),  # pip5/kneeの区切り
            df_joint2.loc[l_rows],
        ],
        axis=0,
    )

    combined_scaled = col_minmax_scale(combined)

    cmap_map = {
        "Tend–1": cyan_cmap,
        "Tend–2": magenta_cmap,
        "Tend–3": blue_cmap,
        "Tend–4": red_cmap,
        "Swel–1": orange_cmap,
        "Swel–2": lime_cmap,
        "Swel–3": gold_cmap,
    }

    def base_from_col(c: str) -> str | None:
        if c == "__blank__":
            return None
        return str(c).rsplit("_", 1)[0] 

    cmap_per_col = []
    for c in combined_scaled.columns:
        base = base_from_col(c)
        cmap_per_col.append(None if base is None else cmap_map.get(base, None))

    n_rows, n_cols = combined_scaled.shape
    fig_h = max(6, 0.45 * n_rows)
    fig_w = max(8, 0.6 * n_cols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("white")

    for i in range(n_rows):
        rowname = combined_scaled.index[i]
        for j in range(n_cols):
            colname = combined_scaled.columns[j]
            v = combined_scaled.iat[i, j]
            cmap = cmap_per_col[j]

            if rowname == "" or colname == "__blank__" or cmap is None:
                continue

            if pd.isna(v):
                ax.add_patch(Rectangle((j, i), 1, 1,
                                       facecolor=nan_grey, edgecolor="black", linewidth=1.2))
            else:
                ax.add_patch(Rectangle((j, i), 1, 1,
                                       facecolor=cmap(float(v)), edgecolor="black", linewidth=1.2))

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(list(combined_scaled.index), fontsize=fontsize)
    ax.tick_params(length=0)

    ax.xaxis.tick_top()
    xticklabels = ["" if c == "__blank__" else c for c in combined_scaled.columns]
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=fontsize, va="bottom")

    label_color_map = {
        "Tend–1": "cyan",
        "Tend–2": "magenta",
        "Tend–3": "blue",
        "Tend–4": "red",
        "Swel–1": "#FF8000",
        "Swel–2": "#7FFF00",
        "Swel–3": "gold",
    }
    for xtick, c in zip(ax.get_xticklabels(), combined_scaled.columns):
        base = base_from_col(c)
        xtick.set_color(label_color_map.get(base, "black") if base is not None else "black")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

    return df_tend, df_swel, df_joint2, combined, combined_scaled


# =========================
# Draw Heatmap
# =========================
df_tender_wide, df_swollen_wide, df_joint, combined_raw, combined_scaled = plot_joint_basis_heatmap(
    df_joint_basis_tender=df_joint_basis,                # Tenderness
    df_joint_basis_swollen=df_joint_basis_swollen,       # Swelling
    savepath="joint_basis_heatmap_TenderRank4_SwollenRank3_RL.pdf",
    fontsize=24,
    nan_grey="lightgrey",
)



