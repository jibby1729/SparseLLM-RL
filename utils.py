import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

def get_rank(matrix : torch.Tensor) -> int:
    return torch.linalg.matrix_rank(matrix).item()

def get_effective_rank(matrix: torch.Tensor) -> float:
    stability_eps = 1e-8
    singular_values, _ = torch.sort(torch.linalg.svdvals(matrix), descending=True)
    dist = singular_values / torch.sum(singular_values)
    # Vectorized entropy computation - much faster than Python loop
    mask = dist > stability_eps
    entropy = -torch.sum(dist[mask] * torch.log(dist[mask]))
    return torch.exp(entropy).item()


def main():
    samples = 100000
    matrix_list = [torch.randn(100, 100) for _ in range(samples)]
    effective_rank_list = [get_effective_rank(matrix) for matrix in matrix_list]
    classic_rank_list = [get_rank(matrix) for matrix in matrix_list]
    plt.hist(effective_rank_list, bins=20, label="Effective Rank", alpha=0.7)
    plt.hist(classic_rank_list, bins=20, label="Classic Rank", alpha=0.7)
    plt.legend()
    plt.title("Effective Rank vs Classic Rank")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()      

if __name__ == "__main__":
    main()