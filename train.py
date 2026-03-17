"""
Autoresearch LoRA fine-tuning script for Qwen3-8B-4bit on Apple Silicon.
Adapted from Karpathy's autoresearch to optimize LoRA hyperparameters.

The autoresearch agent modifies ONLY the hyperparameters block below,
then runs this script. Each experiment is time-budgeted to 5 minutes
of training. The agent keeps improvements and reverts failures.

Usage: ~/.venvs/mlx/bin/python train.py
"""

import math
import os
import time
import types
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from mlx_lm import load as load_model
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, evaluate, iterate_batches
from mlx_lm.tuner.trainer import grad_checkpoint as apply_grad_checkpoint
from mlx_lm.tuner.utils import linear_to_lora_layers

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — this is what autoresearch tunes)
# ---------------------------------------------------------------------------

# LoRA configuration
LORA_RANK = 8
LORA_SCALE = 16.0
LORA_DROPOUT = 0.05
NUM_LORA_LAYERS = -1  # how many layers from the end get LoRA; -1 = all
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

# Training
LEARNING_RATE = 5e-5
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
MAX_SEQ_LENGTH = 2048
MASK_PROMPT = True
GRAD_CHECKPOINT = True

# Optimizer: "adam", "adamw", or "adafactor"
OPTIMIZER = "adamw"
WEIGHT_DECAY = 0.01  # only used with adamw

# LR schedule
WARMUP_RATIO = 0.0  # fraction of budget spent warming up
WARMDOWN_RATIO = 0.3  # fraction of budget spent cooling down
FINAL_LR_FRAC = 0.1  # LR at end of warmdown, as fraction of peak

# Eval
VAL_BATCHES = 25

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------
MODEL_PATH = (
    "/Users/develo/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit"
    "/snapshots/545dc4251c05440727734bcd94334791f6ab0192"
)
DATA_DIR = "/Users/develo/Projects/ailab1/data"
TIME_BUDGET = 300  # 5 minutes training (wall clock, excludes startup/eval)
SEED = 42
STARTUP_EXCLUDE_STEPS = 1  # exclude first N steps from time budget (compilation)

# ---------------------------------------------------------------------------
# LR schedule helper
# ---------------------------------------------------------------------------


def get_lr_multiplier(progress):
    """Warmup → constant → warmdown schedule."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(SEED)

# 1. Load model and tokenizer
print("Loading model...")
model, tokenizer = load_model(MODEL_PATH, tokenizer_config={"trust_remote_code": True})
t_loaded = time.time()
print(f"Model loaded in {t_loaded - t_start:.1f}s")

# 2. Freeze base model, inject LoRA layers
model.freeze()
lora_config = {
    "rank": LORA_RANK,
    "scale": LORA_SCALE,
    "dropout": LORA_DROPOUT,
    "keys": LORA_KEYS,
}
linear_to_lora_layers(model, NUM_LORA_LAYERS, lora_config)

trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
total_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(
    f"Trainable parameters: {trainable_params / total_params:.3%} "
    f"({trainable_params / 1e6:.3f}M / {total_params / 1e6:.3f}M)"
)

# 3. Load datasets
data_config = types.SimpleNamespace(
    mask_prompt=MASK_PROMPT,
    prompt_feature="prompt",
    completion_feature="completion",
    text_feature="text",
    chat_feature="messages",
)
train_raw, valid_raw, _ = load_local_dataset(Path(DATA_DIR), tokenizer, data_config)
train_set = CacheDataset(train_raw)
valid_set = CacheDataset(valid_raw)
print(f"Train: {len(train_set)} examples, Valid: {len(valid_set)} examples")

# 4. Set up optimizer
if OPTIMIZER == "adam":
    opt = optim.Adam(learning_rate=LEARNING_RATE)
elif OPTIMIZER == "adamw":
    opt = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "adafactor":
    opt = optim.Adafactor(learning_rate=LEARNING_RATE)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

# 5. Apply gradient checkpointing
if GRAD_CHECKPOINT and hasattr(model, "layers") and len(model.layers) > 0:
    apply_grad_checkpoint(model.layers[0])

# 6. Compile the training step
loss_value_and_grad = nn.value_and_grad(model, default_loss)

state = [model.state, opt.state, mx.random.state]


@partial(mx.compile, inputs=state, outputs=state)
def compiled_step(batch, lengths):
    (loss, ntoks), grads = loss_value_and_grad(model, batch, lengths)
    grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
    opt.update(model, grads)
    return loss, ntoks


# 7. Training loop (time-budgeted)
model.train()
batch_iter = iterate_batches(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    max_seq_length=MAX_SEQ_LENGTH,
    loop=True,
)

t_data = time.time()
print(f"Setup complete in {t_data - t_start:.1f}s, starting training...")
print(f"Time budget: {TIME_BUDGET}s | Grad accum: {GRAD_ACCUM_STEPS}")

total_training_time = 0.0
step = 0
total_tokens = 0
smooth_loss = 0.0
t_compiled = None

while True:
    t0 = time.time()

    # Gradient accumulation
    accum_loss = 0.0
    accum_ntoks = 0

    for micro in range(GRAD_ACCUM_STEPS):
        batch, lengths = next(batch_iter)

        if GRAD_ACCUM_STEPS == 1:
            # Single step — use compiled version
            loss, ntoks = compiled_step(batch, lengths)
            mx.eval(state)
        else:
            # Manual accumulation — can't use mx.compile easily
            (loss, ntoks), grads = loss_value_and_grad(model, batch, lengths)
            mx.eval(loss, ntoks, grads)
            if micro == 0:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)

        accum_loss += loss.item() * ntoks.item()
        accum_ntoks += ntoks.item()

    if GRAD_ACCUM_STEPS > 1:
        # Average gradients and apply
        accum_grads = tree_map(lambda g: g * (1.0 / GRAD_ACCUM_STEPS), accum_grads)
        opt.update(model, accum_grads)
        mx.eval(model.parameters(), opt.state)

    if t_compiled is None:
        t_compiled = time.time()
        print(f"First step compiled in {t_compiled - t_data:.1f}s")

    dt = time.time() - t0
    if step >= STARTUP_EXCLUDE_STEPS:
        total_training_time += dt

    step_loss = accum_loss / max(accum_ntoks, 1)
    total_tokens += accum_ntoks

    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * step_loss
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    opt.learning_rate = LEARNING_RATE * lrm

    pct_done = 100 * progress
    remaining = max(0.0, TIME_BUDGET - total_training_time)
    tok_per_sec = int(accum_ntoks / dt) if dt > 0 else 0

    print(
        f"\rstep {step:04d} ({pct_done:.1f}%) | loss: {debiased_loss:.4f} | "
        f"lr: {LEARNING_RATE * lrm:.2e} | dt: {dt * 1000:.0f}ms | "
        f"tok/s: {tok_per_sec:,} | remaining: {remaining:.0f}s    ",
        end="",
        flush=True,
    )

    step += 1
    if step >= STARTUP_EXCLUDE_STEPS and total_training_time >= TIME_BUDGET:
        break

print()
t_train = time.time()
print(f"Training completed: {step} steps in {t_train - t_compiled:.1f}s")

# 8. Final evaluation
print("Starting final eval...")
model.eval()
val_loss = evaluate(
    model=model,
    dataset=valid_set,
    batch_size=BATCH_SIZE,
    num_batches=VAL_BATCHES,
    max_seq_length=MAX_SEQ_LENGTH,
    loss=default_loss,
    iterate_batches=iterate_batches,
)
t_eval = time.time()
print(f"Final eval completed in {t_eval - t_train:.1f}s")

# 9. Output in autoresearch format
val_bpb = val_loss / math.log(2)  # nats-per-token → bits-per-token
peak_vram_mb = mx.get_peak_memory() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_eval - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      0.00")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {trainable_params / 1e6:.1f}")
print(f"lora_rank:        {LORA_RANK}")
