# SparseLLM-RL

Repository implementing the paper "Reinforcement Learning Finetunes Small Subnetworks in Large Language Models" (https://arxiv.org/abs/2505.11711) and exploring some additional hypotheses the paper didn't explore.

## ⚠️ Important: TRL Library Version

**This project uses TRL version 0.8.6, which is an older version of the library.** The codebase is specifically designed to work with this version, and newer versions may have breaking API changes. **DO NOT upgrade TRL automatically** - use the exact version specified in `requirements.txt`.

## Setup Instructions

### Prerequisites
- Conda (Miniconda or Anaconda)
- Python 3.8+ (recommended: Python 3.10 or 3.11)

### Installation Steps

1. **Create a new conda environment:**
   ```bash
   conda create -n sparsellm-rl python=3.10
   conda activate sparsellm-rl
   ```

2. **Install PyTorch with CUDA support (if using GPU):**
   ```bash
   # For CUDA 12.1 (adjust version as needed)
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
   
   Or for CPU-only:
   ```bash
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

3. **Install requirements with exact versions (no upgrades):**
   ```bash
   pip install --no-deps -r requirements.txt
   pip install -r requirements.txt --no-upgrade
   ```
   
   The `--no-upgrade` flag ensures pip will not automatically upgrade packages to newer versions. This is critical for maintaining compatibility with TRL 0.8.6.

4. **Verify installation:**
   ```bash
   python -c "import trl; print(f'TRL version: {trl.__version__}')"
   ```
   
   You should see: `TRL version: 0.8.6`

## Project Structure

### Main Scripts

#### `main_ppo.py`
The main training script that implements PPO (Proximal Policy Optimization) for fine-tuning GPT-2 models on sentiment tasks. This script:
- Loads GPT-2-medium as the policy model
- Uses a DistilBERT-based reward model for sentiment classification
- Trains the model using PPO with IMDB review prompts
- Logs rank statistics and weight update metrics to Weights & Biases
- Saves model checkpoints every 20 steps in the `checkpoints/` directory
- Tracks sparse subnetworks by monitoring weight changes above a threshold (tau)

**Key features:**
- Adaptive KL penalty control
- Reward normalization and whitening
- Sparse subnetwork tracking via weight deltas
- Layer-wise rank statistics (classical, effective, stable ranks)

#### `analyze_sparse_subnetwork.py`
Analysis script that examines the sparse subnetworks formed during training. This script:
- Loads checkpoints saved during training
- Computes active weight indices for different tolerance thresholds (tau values)
- Analyzes coverage of the final sparse subnetwork across training steps
- Generates plots showing:
  - Coverage curves: How much of the final subnetwork is present at each step
  - Density curves: Percentage of weights updated at each step
- Saves plots to the `plots/` directory

**Usage:**
```bash
python analyze_sparse_subnetwork.py
```

**Output:**
- `plots/coverage_combined.png`: Coverage of final subnetwork vs training step
- `plots/density_combined.png`: Global density (% updated weights) vs training step

#### `utils.py`
Utility module containing helper functions for rank analysis:

- **`get_rank()`**: Computes classical matrix rank (number of linearly independent columns/rows)

- **`get_effective_rank()`**: Computes effective rank using entropy of singular values. For a matrix $A$ with singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_n$, the effective rank is defined as:
  
  $$p_i = \frac{\sigma_i}{\sum_{j=1}^n \sigma_j}, \quad \text{entropy}(p) = -\sum_{i=1}^n p_i \log(p_i), \quad \text{effective\_rank}(A) = \exp(\text{entropy}(p))$$
  
  The effective rank measures how "spread out" the singular values are. A matrix with uniform singular values has effective rank equal to the number of singular values, while a matrix with one dominant singular value has effective rank close to 1.

- **`get_stable_rank()`**: Computes stable rank (also known as numerical rank). For a matrix $A$ with singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_n$:
  
  $$\text{stable\_rank}(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_{i=1}^n \sigma_i^2}{\sigma_1^2}$$
  
  where $\|A\|_F$ is the Frobenius norm and $\|A\|_2 = \sigma_1$ is the spectral norm (largest singular value). The stable rank is always between 1 and the classical rank, and provides a more robust measure of rank in the presence of numerical errors.

- **`get_layer_rank_stats()`**: Computes comprehensive statistics for model layers including:
  - Classical, effective, and stable ranks of weight deltas ($\Delta W = W_t - W_0$)
  - Frobenius norm of weight changes
  - Percentage of weights updated above threshold
  - Fraction of large weight updates

**GPT-2 Layers Analyzed:**

The analysis focuses on 2D weight matrices from the following GPT-2 components:

- **Attention layers** (`transformer.h[i].attn`):
  - `c_attn.weight`: Input projection matrix combining Query, Key, and Value projections (shape: `[embed_dim, 3*embed_dim]`)
  - `c_proj.weight`: Output projection matrix (shape: `[embed_dim, embed_dim]`)

- **MLP layers** (`transformer.h[i].mlp`):
  - `c_fc.weight`: Feedforward layer weight matrix (shape: `[embed_dim, 4*embed_dim]` for GPT-2-medium)

- **Language model head**:
  - `lm_head.weight`: Output projection to vocabulary (shape: `[vocab_size, embed_dim]`)

These are the primary weight matrices that undergo updates during fine-tuning. Layer normalization and bias parameters are excluded from the analysis.

### Utility Scripts

#### `weight_shapes.py`
Simple utility script to inspect weight tensor shapes in GPT-2-medium model. Useful for understanding model architecture.

#### `plots/layer_ranks.py`
Script for generating layer rank visualizations (if present).

## Output Directories

- **`checkpoints/`**: Model checkpoints saved during training (every 20 steps)
  - Format: `step_XXX.pt` where XXX is the step number (000, 020, 040, ..., 600)
  
- **`plots/`**: All generated plots and visualizations
  - Coverage and density plots from `analyze_sparse_subnetwork.py`
  - Rank statistics plots
  - Other analysis visualizations

- **`ppo-gpt2-medium-sentiment-final/`**: Final trained model saved after training completes

## Usage Example

1. **Train the model:**
   ```bash
   python main_ppo.py
   ```
   
   This will:
   - Create checkpoints in `checkpoints/` directory
   - Log metrics to Weights & Biases (if enabled)
   - Save final model to `ppo-gpt2-medium-sentiment-final/`

2. **Analyze sparse subnetworks:**
   ```bash
   python analyze_sparse_subnetwork.py
   ```
   
   This will generate plots in the `plots/` directory showing how sparse subnetworks evolve during training.

## Configuration

Key parameters in `main_ppo.py` can be adjusted via the `ExpConfig` dataclass:
- `policy_name`: Base language model (default: "gpt2-medium")
- `reward_model_name`: Reward model for sentiment (default: "lvwerra/distilbert-imdb")
- `tau`: Tolerance threshold for "updated" weights (default: 1e-6)
- `num_updates`: Number of PPO training steps (default: 600)
- `save_ckpt_every`: Checkpoint frequency (default: 20)

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages:
- `torch==2.9.1`: PyTorch
- `transformers==4.57.1`: Hugging Face Transformers
- **`trl==0.8.6`**: TRL library (⚠️ OLD VERSION - DO NOT UPGRADE)
- `datasets==4.4.1`: Hugging Face Datasets
- `wandb==0.23.0`: Weights & Biases for logging
- `numpy`, `scipy`, `matplotlib`: Scientific computing and plotting

## Notes

- The code uses `torch.bfloat16` for mixed precision training
- CUDA is required for GPU training (set `device: str = "cuda"` in config)
- Weights & Biases logging can be disabled by setting `use_wandb: bool = False`
- Checkpoints are saved as PyTorch state dictionaries (`.pt` files)
