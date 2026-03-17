# autoresearch-mlx — LoRA Fine-Tuning Edition

This is an adapted version of Karpathy's autoresearch for **LoRA fine-tuning optimization** on Apple Silicon. Instead of training a small GPT from scratch, it autonomously searches for optimal LoRA hyperparameters for Qwen3-8B-4bit fine-tuning.

**Monorepo note:** This project may live inside a larger repo. Always stage only `autoresearch-mlx/` paths. Never use blind `git add -A`.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — the file you modify. LoRA config, optimizer, training hyperparameters.
4. **Verify data exists**: Check that `/Users/develo/Projects/ailab1/data/train.jsonl` and `valid.jsonl` exist. These are the OpenHermes 2.5 dataset formatted for mlx-lm.
5. **Initialize results.tsv**: Create `results.tsv` with header row and baseline entry. Run `~/.venvs/mlx/bin/python train.py` once to establish YOUR baseline on this hardware. Do NOT use baseline numbers from other runs.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Silicon via MLX. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it as:

```
~/.venvs/mlx/bin/python train.py
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything in the hyperparameters block is fair game:
  - **LoRA architecture**: rank, scale (alpha), dropout, target modules, number of LoRA layers
  - **Training**: learning rate, batch size, gradient accumulation, max sequence length, prompt masking
  - **Optimizer**: adam, adamw, adafactor; weight decay
  - **LR schedule**: warmup ratio, warmdown ratio, final LR fraction
  - **Eval**: number of validation batches
- You may also modify the training loop logic itself (e.g., add gradient clipping, change schedule shape, etc.)

**What you CANNOT do:**
- Change the model (Qwen3-8B-4bit) or the data files (train.jsonl, valid.jsonl).
- Install new packages or add dependencies.
- Modify the constants block (MODEL_PATH, DATA_DIR, TIME_BUDGET, SEED).
- Change the output format (the `---` block at the end must stay exactly as-is).

**The goal is simple: get the lowest val_bpb (lower = better).** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything in the hyperparameters block is fair game.

**Memory** is a soft constraint. MLX uses unified memory (48GB on this machine). The base model uses ~4.3GB. Training overhead depends on batch size, sequence length, and LoRA rank. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up past ~30GB peak.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

## Experiment ideas

Here are productive directions to explore (roughly ordered by expected impact):

### High impact
- **Learning rate**: Try 1e-5 to 5e-4 range. This is usually the single biggest knob.
- **LoRA rank**: Try 4, 8, 16, 32, 64. Higher rank = more capacity but slower.
- **Number of LoRA layers**: Try 8, 16, 24, 32, or -1 (all). More layers = more trainable params.
- **Sequence length**: Shorter (512, 1024) means more steps in 5 minutes. Sometimes more steps at shorter context beats fewer steps at full context.

### Medium impact
- **Target modules**: Try attention-only (q/k/v/o_proj) vs attention+MLP (add gate/up/down_proj). MLP modules add more params but may help.
- **Batch size / grad accumulation**: Larger effective batch (e.g., batch=2 × accum=8) vs smaller (batch=4 × accum=1). Tradeoff between gradient quality and steps per budget.
- **Optimizer**: adamw with weight_decay=0.01-0.1 vs plain adam. Adafactor for lower memory.
- **Scale (alpha)**: Try scale = rank (standard), scale = rank*2, or scale = rank/2.

### Lower impact (but worth trying)
- **LR schedule**: Warmup (0.05-0.1) + warmdown (0.2-0.5). Different final_lr_frac values.
- **Dropout**: 0.0 vs 0.05 vs 0.1. May help with overfitting on small val sets.
- **Prompt masking**: On vs off. With masking, loss only computed on assistant turns.
- **Gradient clipping**: Change max_norm or remove it entirely.

### Radical ideas
- **Two-phase training**: High LR for first 70%, low LR for last 30%.
- **Progressive unfreezing**: Start with fewer LoRA layers, add more mid-training.
- **Very small rank + all layers** vs **large rank + few layers**.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.876543
training_seconds: 300.1
total_seconds:    345.7
peak_vram_mb:     18500.2
mfu_percent:      0.00
total_tokens_M:   0.4
num_steps:        85
num_params_M:     19.4
lora_rank:        16
```

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 0.876543) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 18.5 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.920000	17.7	keep	baseline (rank=16 lr=1e-4 layers=16)
e4f5g6h	0.876543	17.7	keep	lower lr to 5e-5
i7j8k9l	0.890123	18.2	discard	rank=64 (worse despite more params)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the hyperparameters (or training loop).
3. `git add autoresearch-mlx/train.py && git commit -m "experiment: <description>"` (never `git add -A` — this may be inside a larger repo)
4. Run the experiment: `~/.venvs/mlx/bin/python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), `git add autoresearch-mlx/results.tsv && git commit --amend --no-edit` to include the log, advancing the branch
9. If val_bpb is equal or worse, record the discard commit hash, then `git reset --hard <previous kept commit>` to discard it cleanly

**Timeout**: Each experiment should take ~8 minutes total (5 min training + ~30s model load + ~1 min compile + ~1 min eval). If a run exceeds 15 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, or a bug), use your judgment: fix easy bugs, skip fundamentally broken ideas.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

Overnight (~8 hours) at ~8 min per experiment ≈ 60 experiments. The user wakes up to a results.tsv with the best LoRA hyperparameters found.
