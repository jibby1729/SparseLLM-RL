import os
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import torch
import torch.nn as nn
from dataclasses import dataclass
from datasets import load_dataset
import wandb
from utils import *
from typing import List, Dict
import numpy
import scipy
import trl
import transformers

print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("trl", trl.__version__)
print("transformers", transformers.__version__)

@dataclass
class ExpConfig:
    policy_name: str = "gpt2-medium"   # base LM
    reward_model_name: str = "lvwerra/distilbert-imdb"
    max_prompt_tokens: int = 128
    max_new_tokens: int = 64
    prompt_prefix_words: int = 8           # use only the first N words of each IMDB review
    target_sentiment: str = "positive review"     # target sentiment for generation

    batch_size: int = 16              # PPO batch per update
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    learning_rate: float = 2e-6       # reduced for stability
    target_kl: float = 0.01
    cliprange: float = 0.2            # PPO clipping range
    num_updates: int = 200                            # PPO updates

    rank_log_every: int = 20                          # log rank stats every N steps
    tau: float = 1e-4                                  # tolerance for "updated" weights

    save_ckpt_every: int = 10                         # save model every N steps
    ckpt_dir: str = "checkpoints"

    device: str = "cuda"

    dtype: torch.dtype = torch.float32 
    use_wandb: bool = False             # toggle W&B logging


cfg = ExpConfig()


def load_imdb_dataset(max_samples: int = 10000) -> List[str]:
    ds = load_dataset("imdb", split="train")
    texts = ds[:max_samples]["text"]
    random.shuffle(texts)

    return texts

def build_policy_and_ref_model() -> tuple[AutoTokenizer, nn.Module, nn.Module]:
    dtype = cfg.dtype
    tok = AutoTokenizer.from_pretrained(cfg.policy_name, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(cfg.device)
    policy.config.use_cache = False
    policy.gradient_checkpointing_disable()

    ref = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(cfg.device)
    ref.config.use_cache = False
    ref.requires_grad_(False)
    ref.eval()

    return tok, policy, ref

def reward_model() -> tuple[AutoTokenizer, nn.Module]:
    rm_tok = AutoTokenizer.from_pretrained(cfg.reward_model_name)
    rm = AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model_name, 
        torch_dtype=cfg.dtype, 
        use_safetensors=True
    ).to(cfg.device)
    rm.eval()
    return rm_tok, rm

@torch.no_grad()
def compute_rewards(rm_tok, rm, text: List[str]) -> torch.Tensor:
    """ Reward = logit(pos) - logit(neg) """
    enc = rm_tok(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(cfg.device)
    logits = rm(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    reward = logits[:, 1] - logits[:, 0]
    return reward.detach().to(torch.float32)


def maybe_save_checkpoint(step: int, policy: nn.Module):
    if step % cfg.save_ckpt_every != 0:
        return
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"step_{step:03d}.pt")
    torch.save(policy.state_dict(), path)
    print(f"[ckpt] saved {path}")

def main():
    torch.manual_seed(0)
    random.seed(0)
    print(f"okayyyy now Using device: {cfg.device}")

    if cfg.use_wandb:
        os.environ.pop("WANDB_DISABLED", None)
        wandb.init(project="ppo-gpt2small-sentiment", config=vars(cfg))
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # models
    tok, policy, ref = build_policy_and_ref_model()
    rm_tok, rm = reward_model()

    # PPO config (TRL 0.8.x API)
    # Note: In TRL 0.8.6, adaptive KL control is enabled by default when target_kl is set
    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        target_kl=cfg.target_kl,  # This enables adaptive KL penalty control
        cliprange=cfg.cliprange,
        remove_unused_columns=False,
    )

    print('set ppo config')

    trainer = PPOTrainer(
        ppo_config,   # config (positional)
        policy,       # model
        ref,          # reference model
        tok,          # tokenizer
    )

    print('set trainer')


    # initial state for ΔW
    init_state: Dict[str, torch.Tensor] = {
        name: p.detach().cpu().clone()
        for name, p in policy.named_parameters()
        if p.requires_grad
    }

    print('set init state')

    # save ckpt at step 0 (for ΔW baseline, and later sparse-subnetwork analysis)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(trainer.model.state_dict(), os.path.join(cfg.ckpt_dir, "step_000.pt"))

    # data
    prompts = load_imdb_dataset()
    print(f"Loaded {len(prompts)} IMDB prompts")

    def sample_batch(bs: int) -> List[str]:
        return random.sample(prompts, bs)

    print('starting ppo loop...')
    # PPO loop
    for step in range(1, cfg.num_updates + 1):
        # ---- sample prompts ----
        batch_texts = sample_batch(cfg.batch_size)
        batch_prefixes = [" ".join(t.split()[:cfg.prompt_prefix_words]) for t in batch_texts]
        
        # Add sentiment instruction to prompts for generation
        # Format: "Sentiment: positive\nReview: <truncated_review>"
        batch_prompts_with_sentiment = [
            f"Sentiment: {cfg.target_sentiment}. {prefix}" 
            for prefix in batch_prefixes
        ]

        enc = tok(
            batch_prompts_with_sentiment,  # use prompts with sentiment instruction
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_prompt_tokens,
        ).to(trainer.accelerator.device)

        input_ids = enc["input_ids"]  # [B, Lmax]
        attn = enc["attention_mask"]  # [B, Lmax]
        lengths = attn.sum(dim=1).tolist()  # true prompt length per sample

        # Build list of trimmed query tensors for TRL (right padding: prompt is at the start)
        query_tensors = [input_ids[i, :lengths[i]].clone() for i in range(len(lengths))]

        # ---- generate with current policy ----
        print(f"[step {step}] sampling responses...") if step % 5 == 0 else None
        gen_tensors = trainer.generate(
            query_tensors,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=24,  # enforce minimum tokens to ensure substantial completions
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1,
            pad_token_id=tok.eos_token_id,
        )  # list of [L+L_new] tensors
        print(f"[step {step}] generated {len(gen_tensors) if isinstance(gen_tensors, list) else gen_tensors.size(0)} sequences.") if step % 5 == 0 else None

        # Extract response portion from each generated sequence
        # With right padding, prompt is at the start, so slice at each item's true length
        response_list = []
        valid_indices = []  # track which samples have valid (non-empty) responses
        for idx, (gen_seq, prompt_len) in enumerate(zip(gen_tensors, lengths)):
            response = gen_seq[prompt_len:]  # slice off the prompt, keep only response
            # Filter out empty responses - don't create synthetic tokens
            if len(response) > 0:
                response_list.append(response)
                valid_indices.append(idx)

        # Check if we have any valid responses
        if len(response_list) == 0:
            print(f"[step {step}] all responses were empty; skipping step.")
            continue
        
        # Log average response length
        avg_len = float(torch.tensor([r.numel() for r in response_list]).float().mean())
        print(f"[step {step}] avg_response_len={avg_len:.1f}") if step % 5 == 0 else None
        
        # Filter query_tensors and batch data to match valid responses
        query_tensors_filtered = [query_tensors[i] for i in valid_indices]
        lengths_filtered = [lengths[i] for i in valid_indices]
        batch_texts_filtered = [batch_texts[i] for i in valid_indices]
        batch_prefixes_filtered = [batch_prefixes[i] for i in valid_indices]

        # Pad responses and masks consistently for PPO step
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(r.size(0) for r in response_list)
        padded_resps, padded_masks = [], []
        for r in response_list:
            pad_len = max_len - r.size(0)
            if pad_len > 0:
                r = torch.cat([r, torch.full((pad_len,), pad_id, device=r.device, dtype=r.dtype)])
            mask = (r != pad_id).long()
            padded_resps.append(r)
            padded_masks.append(mask)

        # Pad responses for reward computation (using same padding as PPO)
        response_tensors = torch.stack(padded_resps)  # [B, max_len]

        # Decode generated responses
        generated_texts = tok.batch_decode(response_tensors, skip_special_tokens=True)
        
        # For reward computation: use truncated IMDB review + generated text (not sentiment instruction)
        # This ensures the classifier sees the actual review context + continuation
        full_texts_for_reward = [
            batch_prefixes_filtered[i] + " " + generated_texts[i]
            for i in range(len(generated_texts))
        ]

        # ---- compute rewards ----
        rewards_vec = compute_rewards(rm_tok, rm, full_texts_for_reward)  # [B] - truncated review + generated text
        
        # Keep rewards directional: don't center (zero the mean) so positive rewards push policy correctly
        # Only scale down to prevent overly aggressive updates while maintaining direction
        reward_scale = 0.02  # scale factor to keep advantages reasonable
        rewards_scaled = rewards_vec * reward_scale
        
        reward_tensors = [r.unsqueeze(0) for r in rewards_scaled]  # list[Tensor[1]]
        
        # Debugging: print reward stats including max
        raw_max = rewards_vec.max().item()
        print(f"[step {step}] computed rewards, raw_mean={rewards_vec.mean().item():.4f}, raw_std={rewards_vec.std().item():.4f}, raw_max={raw_max:.4f}, scaled_mean={rewards_scaled.mean().item():.4f}, scaled_std={rewards_scaled.std().item():.4f}")
        
        # Debugging: print sample outputs and their rewards (every step, or every N steps)
        if step % 5 == 0 or step <= 3:  # print more frequently in early steps
            sample_idx = 0
            if len(full_texts_for_reward) > sample_idx:
                # Show IMDB prompt and GPT-generated text separately for clarity
                imdb_prompt = batch_prefixes_filtered[sample_idx]
                gpt_generated = generated_texts[sample_idx]
                full_text = full_texts_for_reward[sample_idx]
                
                print(f"[step {step}] === Sample Output ===")
                print(f"[step {step}] IMDB PROMPT (first {cfg.prompt_prefix_words} words): {imdb_prompt}")
                print(f"[step {step}] GPT GENERATED TEXT: {gpt_generated[:400]}{'...' if len(gpt_generated) > 400 else ''}")
                print(f"[step {step}] Reward: raw={rewards_vec[sample_idx].item():.4f}, scaled={rewards_scaled[sample_idx].item():.4f}")
                print(f"[step {step}] ====================")

        # ---- PPO update ----
        # Use filtered query_tensors with padded responses and masks
        stats = trainer.step(query_tensors_filtered, padded_resps, reward_tensors, padded_masks)
        
        # Debugging: print KL and ratio stats
        def safe_get_float(key, default=float('nan')):
            val = stats.get(key, default)
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else float(val.mean().item())
            elif hasattr(val, '__len__') and not isinstance(val, str):  # array-like (numpy, list, etc.)
                try:
                    if isinstance(val, numpy.ndarray):
                        return float(val.item() if val.size == 1 else val.mean())
                except:
                    pass
                try:
                    # Try to get mean if it's an array
                    return float(sum(val) / len(val)) if len(val) > 0 else default
                except:
                    return default
            else:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
        
        kl_val = safe_get_float("objective/kl")
        kl_dist_val = safe_get_float("objective/kl_dist")
        print(f"[step {step}] ppo step done. KL={kl_val:.4f}, KL_dist={kl_dist_val:.4f}, stats keys: {list(stats.keys())[:5]} ...") if step % 5 == 0 else None
        
        # Debugging: print response length histogram (min/max)
        response_lens = [r.numel() for r in response_list]
        if response_lens:
            min_len, max_len = min(response_lens), max(response_lens)
            print(f"[step {step}] response lengths: min={min_len}, max={max_len}, avg={avg_len:.1f}") if step % 5 == 0 else None

        # log PPO stats every 10 steps
        if step % 10 == 0:
            decoded_responses = tok.batch_decode(response_tensors, skip_special_tokens=True)
            log_batch = {"query": batch_texts_filtered, "response": decoded_responses}
            trainer.log_stats(
                stats=stats,
                batch=log_batch,
                rewards=rewards_vec,
            )

        # ---- rank + sparsity logging ----
        if step % cfg.rank_log_every == 0:
            # get_layer_rank_stats should accept (model, init_state, tau=...)
            layer_stats = get_layer_rank_stats(trainer.model, init_state, tolerance=cfg.tau)

            # log a few representative layers (avoid spamming W&B)
            if cfg.use_wandb and wandb.run is not None:
                for name, s in list(layer_stats.items())[:10]:
                    wandb.log(
                        {
                            f"rank/eff/{name}": s["eff_rank"],
                            f"rank/stable/{name}": s["stable_rank"],
                            f"norm/fro/{name}": s["fro_norm"],
                            f"sparsity/frac_large/{name}": s["frac_large"],
                            f"sparsity/percent_updated/{name}": s["percent_updated"],
                        },
                        step=step,
                    )

            # diagnostic print
            some_name = sorted(layer_stats.keys())[0]
            print(
                f"[step {step}] {some_name}: "
                f"eff={layer_stats[some_name]['eff_rank']:.1f}, "
                f"stable={layer_stats[some_name]['stable_rank']:.1f}, "
                f"fro={layer_stats[some_name]['fro_norm']:.2e}, "
                f"percent_updated={layer_stats[some_name]['percent_updated']:.3f}, "
                f"frac_large={layer_stats[some_name]['frac_large']:.3f}"
            )

        # ---- save checkpoints for sparse-subnetwork analysis ----
        maybe_save_checkpoint(step, trainer.model)

    # final save
    trainer.save_pretrained(f"ppo-{cfg.policy_name}-sentiment-final")
    print("Training finished, model saved.")



if __name__ == "__main__":
    main()
