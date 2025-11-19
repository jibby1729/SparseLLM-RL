# analyze_sparse_subnetwork.py

import os
import gc
from typing import Dict, List
import torch
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = "gpt2-medium"
CKPT_DIR = "checkpoints"

# which steps you saved; adjust if different
# Checkpoints are saved every 20 steps, up to step 600
STEPS = list(range(0, 601, 20))   # 0, 20, 40, ..., 600
TAUS = [1e-7, 1e-6, 1e-5]          # Tolerances to analyze

def load_state(step: int) -> Dict[str, torch.Tensor]:
    """Load checkpoint state dict for a given step."""
    path = os.path.join(CKPT_DIR, f"step_{step:03d}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    sd = torch.load(path, map_location="cpu")
    return sd

def collect_tracked_names(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """
    Collect parameter names to track, matching the filter used in get_layer_rank_stats.
    Only tracks 2D weight matrices from attention/MLP layers.
    """
    name_filter = ("attn", "mlp", "c_attn", "c_proj", "c_fc", "lm_head")
    names = []
    for name, tensor in state_dict.items():
        if not name.endswith("weight"):
            continue
        if not any(tok in name for tok in name_filter):
            continue
        if tensor.ndim != 2:
            continue
        names.append(name)
    return sorted(names)  # sort for consistency

def compute_active_indices(
    base_sd: Dict[str, torch.Tensor],
    cur_sd: Dict[str, torch.Tensor],
    tracked_names: List[str],
    tau: float,
) -> set:
    """
    Compute indices of weights with |ΔW| > tau WITHOUT building the full concatenated vector.
    Returns a set of global indices (accounting for offsets from each layer).
    This is memory-efficient: processes one layer at a time and only stores indices.
    """
    active_indices = set()
    offset = 0
    
    for name in tracked_names:
        W0 = base_sd[name].to(torch.float32)
        Wt = cur_sd[name].to(torch.float32)
        dW_abs = (Wt - W0).abs().view(-1)  # Flatten to 1D
        
        # Find active indices in this layer
        mask = (dW_abs > tau)
        local_indices = torch.nonzero(mask, as_tuple=False).view(-1)
        
        # Convert to global indices by adding offset
        global_indices = (local_indices + offset).tolist()
        active_indices.update(global_indices)
        
        # Update offset for next layer
        offset += dW_abs.numel()
        
        # Free memory immediately
        del W0, Wt, dW_abs, mask, local_indices
    
    return active_indices

def main():
    print("=" * 60)
    print("Sparse Subnetwork Analysis for Multiple Tolerances")
    print("=" * 60)
    
    # 1. load base checkpoint (once)
    print(f"\nLoading base checkpoint from {CKPT_DIR}...")
    base_sd = load_state(STEPS[0])
    print(f"✓ Loaded base step {STEPS[0]}")
    
    # 2. decide which parameters to track
    tracked_names = collect_tracked_names(base_sd)
    N_total = sum(base_sd[name].numel() for name in tracked_names)
    print(f"✓ Tracking {len(tracked_names)} parameter tensors ({N_total:,} total weights)")
    if len(tracked_names) == 0:
        print("✗ No parameters tracked, exiting.")
        return

    all_results = {}

    for tau in TAUS:
        print(f"\n{'─'*20} Analyzing for tau = {tau:.1e} {'─'*20}")
        
        # 3. Define final sparse subnetwork for this tau
        final_sd = load_state(STEPS[-1])
        set_final = compute_active_indices(base_sd, final_sd, tracked_names, tau)
        N_final = len(set_final)
        sparsity_final = N_final / N_total if N_total > 0 else 0.0
        
        print(f"  Final subnetwork size: {N_final:,} weights ({100 * sparsity_final:.4f}%)")
        
        # Free final state dict before looping
        del final_sd
        gc.collect()

        # 4. Compute coverage and sparsity for each checkpoint
        coverages = []
        sparsities = []
        for step in STEPS:
            try:
                cur_sd = load_state(step)
                set_t = compute_active_indices(base_sd, cur_sd, tracked_names, tau)
                N_t = len(set_t)
                sparsity_t = N_t / N_total if N_total > 0 else 0.0
                inter = len(set_final & set_t)
                coverage_t = inter / N_final if N_final > 0 else 0.0
                
                sparsities.append(sparsity_t)
                coverages.append(coverage_t)
                
                print(f"    Step {step:3d}: sparsity={sparsity_t:.6f}, coverage={coverage_t:.6f}")
                
                del cur_sd, set_t
                gc.collect()
            except FileNotFoundError as e:
                print(f"    ⚠ Step {step:3d}: {e}")
                sparsities.append(float('nan'))
                coverages.append(float('nan'))

        all_results[tau] = {"coverages": coverages, "sparsities": sparsities}

    # 5. Plot combined curves
    print(f"\n{'='*20} Generating Combined Plots {'='*20}")
    os.makedirs("plots", exist_ok=True)
    steps_np = np.array(STEPS, dtype=float)

    # Combined coverage curve
    plt.figure(figsize=(10, 6))
    for tau, results in all_results.items():
        plt.plot(steps_np, results["coverages"], marker="o", linestyle="-", markersize=4, label=f"τ = {tau:.0e}")
    plt.xlabel("PPO Step", fontsize=12)
    plt.ylabel("Coverage of Final Subnetwork", fontsize=12)
    plt.title("Subnetwork Coverage vs. Training Step for Different τ", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.ylim([0, 1.05])
    coverage_path = "plots/coverage_combined.png"
    plt.savefig(coverage_path, dpi=150)
    print(f"✓ Saved combined coverage plot: {coverage_path}")
    plt.close()

    # Combined density curve
    plt.figure(figsize=(10, 6))
    for tau, results in all_results.items():
        # Convert fraction to percentage
        densities_pct = [x * 100 for x in results["sparsities"]]
        plt.plot(steps_np, densities_pct, marker="o", linestyle="-", markersize=4, label=f"τ = {tau:.0e}")
    
    plt.xlabel("PPO Step", fontsize=12)
    plt.ylabel("Global Density (% of Updated Weights)", fontsize=12)
    plt.title("Global Density vs. Training Step for Different τ", fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    density_path = "plots/density_combined.png"
    plt.savefig(density_path, dpi=150)
    print(f"✓ Saved combined density plot: {density_path}")
    plt.close()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

