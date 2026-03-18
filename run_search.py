#!/usr/bin/env python3
"""
Autoresearch orchestrator — uses local Qwen (via Ollama) as the researcher.
Fully autonomous hyperparameter search. No cloud, no Claude, no permissions.

Git workflow (like Karpathy):
  - Each experiment commits experiment_config.json
  - Improvements: commit stays (branch advances)
  - No improvement: git reset --hard HEAD~1 (branch snaps back to last best)
  - Branch tip always = best config found so far

At the end, auto-applies winning config to ailab1/finetune/lora_v1.yaml.

Usage: ~/.venvs/mlx/bin/python run_search.py [--experiments N]
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

OLLAMA_MODEL = "qwen-local"
OLLAMA_URL = "http://localhost:11434/api/chat"
TRAIN_SCRIPT = Path(__file__).parent / "train.py"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
CONFIG_FILE = Path(__file__).parent / "experiment_config.json"
BEST_CONFIG_FILE = Path(__file__).parent / "best_config.json"
LORA_YAML = Path(__file__).parent.parent / "ailab1" / "finetune" / "lora_v1.yaml"
DEFAULT_EXPERIMENTS = 10

BASELINE_CONFIG = {
    "lora_rank": 8,
    "lora_scale": 16.0,
    "lora_dropout": 0.05,
    "num_lora_layers": -1,
    "lora_keys": "attention+mlp",
    "learning_rate": 5e-5,
    "batch_size": 4,
    "grad_accum_steps": 2,
    "max_seq_length": 2048,
    "optimizer": "adamw",
    "weight_decay": 0.01,
    "warmup_ratio": 0.0,
    "warmdown_ratio": 0.3,
    "final_lr_frac": 0.1,
    "mask_prompt": True,
    "grad_checkpoint": True,
}

SEARCH_SPACE_DESC = """Hyperparameters you can change (JSON keys):
- lora_rank: int — 4, 8, 16, 32, 64. Controls adapter capacity.
- lora_scale: float — typically 2x rank. Controls adaptation strength.
- lora_dropout: float — 0.0 to 0.1. Regularization.
- num_lora_layers: int — 8, 16, 24, 32, or -1 (all 36 layers).
- lora_keys: "attention_only" or "attention+mlp". Which modules get LoRA.
- learning_rate: float — 1e-5 to 5e-4. Usually biggest impact.
- batch_size: int — 1, 2, 4, 8. Larger = fewer steps in 5 min.
- grad_accum_steps: int — 1, 2, 4, 8. Effective batch = batch_size * this.
- max_seq_length: int — 512, 1024, 2048. Shorter = more steps.
- optimizer: "adam", "adamw", or "adafactor".
- weight_decay: float — 0.0 to 0.1. Only used with adamw.
- warmup_ratio: float — 0.0 to 0.2. Fraction of time warming up LR.
- warmdown_ratio: float — 0.0 to 0.5. Fraction of time cooling down LR.
- final_lr_frac: float — 0.01 to 0.5. LR at end as fraction of peak.
- mask_prompt: true/false. Whether to mask prompt tokens in loss."""

LORA_KEYS_MAP = {
    "attention_only": [
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
    ],
    "attention+mlp": [
        "self_attn.q_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ],
}

REPO = Path(__file__).parent


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(args, check=True):
    result = subprocess.run(
        ["git"] + args, cwd=REPO,
        capture_output=True, text=True
    )
    if check and result.returncode != 0:
        print(f"  git error: {result.stderr.strip()}")
    return result


def git_commit(description):
    git(["add", "experiment_config.json", "best_config.json"])
    result = git(["commit", "-m", f"experiment: {description}"], check=False)
    if result.returncode != 0:
        return None
    # Return short commit hash
    return git(["rev-parse", "--short", "HEAD"]).stdout.strip()


def git_reset():
    """Discard last commit (experiment was no improvement)."""
    git(["reset", "--hard", "HEAD~1"])


def git_current_hash():
    return git(["rev-parse", "--short", "HEAD"]).stdout.strip()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def read_results():
    if RESULTS_FILE.exists():
        return RESULTS_FILE.read_text()
    return "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"


def get_best_bpb():
    best = float("inf")
    for line in RESULTS_FILE.read_text().strip().split("\n")[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            try:
                best = min(best, float(parts[1]))
            except ValueError:
                pass
    return best


def log_result(commit, val_bpb, memory_gb, status, description):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n")


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

def load_best_config():
    if BEST_CONFIG_FILE.exists():
        return json.loads(BEST_CONFIG_FILE.read_text())
    return BASELINE_CONFIG.copy()


def save_best_config(config):
    # Store lora_keys as string for readability
    save = config.copy()
    if isinstance(save.get("lora_keys"), list):
        save["lora_keys"] = "attention_only" if len(save["lora_keys"]) <= 4 else "attention+mlp"
    BEST_CONFIG_FILE.write_text(json.dumps(save, indent=2))
    return save


def build_train_config(proposed, base_config):
    config = base_config.copy()
    description = proposed.pop("description", "no description")
    for k, v in proposed.items():
        if k in config:
            config[k] = v
    lora_keys_val = config.get("lora_keys", "attention+mlp")
    if isinstance(lora_keys_val, str):
        for key, modules in LORA_KEYS_MAP.items():
            if key in lora_keys_val.lower().replace(" ", ""):
                config["lora_keys"] = modules
                break
        else:
            config["lora_keys"] = LORA_KEYS_MAP["attention+mlp"]
    return config, description


# ---------------------------------------------------------------------------
# Qwen researcher
# ---------------------------------------------------------------------------

def ask_qwen(results_history, experiment_num, total_experiments, best_config):
    import urllib.request

    prompt = f"""/no_think
You are a machine learning researcher optimizing LoRA fine-tuning hyperparameters for Qwen3-8B.
Each experiment trains for exactly 5 minutes. Goal: minimize val_bpb (lower = better). Memory must stay under 30GB.

## Results so far (lower val_bpb = better):
{results_history}

## Current best config:
{json.dumps(best_config, indent=2)}

{SEARCH_SPACE_DESC}

This is experiment {experiment_num} of {total_experiments}.
Analyze what worked and what didn't. Propose ONE new experiment that you think will beat the current best.

Respond with ONLY a JSON object with the parameters you want to change. Include a "description" field.
Example: {{"learning_rate": 1e-4, "lora_rank": 16, "description": "higher lr with larger rank"}}

JSON only, no other text:"""

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "num_predict": 512},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())

    return data["message"]["content"]


def parse_config(response):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(config):
    CONFIG_FILE.write_text(json.dumps(config, indent=2))

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(CONFIG_FILE)],
        capture_output=True, text=True, timeout=900,
        cwd=REPO,
    )

    output = result.stdout + result.stderr
    (REPO / "run.log").write_text(output)

    val_bpb = None
    peak_mem = None
    for line in output.split("\n"):
        if line.startswith("val_bpb:"):
            val_bpb = float(line.split(":")[1].strip())
        elif line.startswith("peak_vram_mb:"):
            peak_mem = float(line.split(":")[1].strip()) / 1024

    return val_bpb, peak_mem, output


# ---------------------------------------------------------------------------
# Auto-apply best config to lora_v1.yaml
# ---------------------------------------------------------------------------

def apply_to_yaml(config):
    if not LORA_YAML.exists():
        print(f"  lora_v1.yaml not found at {LORA_YAML}, skipping auto-apply")
        return

    lora_keys = config.get("lora_keys")
    if isinstance(lora_keys, str):
        lora_keys = LORA_KEYS_MAP.get(lora_keys, LORA_KEYS_MAP["attention+mlp"])

    keys_yaml = "\n".join(f"    - {k}" for k in lora_keys)
    num_layers = config.get("num_lora_layers", -1)
    # mlx_lm uses total layer count when num_layers=-1 (all layers = 36 for Qwen3-8B)
    num_layers_val = 36 if num_layers == -1 else num_layers

    yaml = f"""# Phase 1 LoRA fine-tune — Qwen3-8B-4bit on OpenHermes 2.5
# Run from project root:
#   ~/.venvs/mlx/bin/mlx_lm.lora -c finetune/lora_v1.yaml
# AUTO-GENERATED by autoresearch/run_search.py — best config from search

# Keep in sync with config.py MODEL_PATH
model: /Users/develo/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/545dc4251c05440727734bcd94334791f6ab0192
data: data/
train: true

# LoRA adapter output
adapter_path: models/adapters/v1

# Training
iters: 1000
batch_size: {config.get('batch_size', 4)}
learning_rate: {config.get('learning_rate', 5e-5)}
mask_prompt: {str(config.get('mask_prompt', True)).lower()}
grad_checkpoint: {str(config.get('grad_checkpoint', True)).lower()}
grad_accumulation_steps: {config.get('grad_accum_steps', 2)}

# LoRA config
num_layers: {num_layers_val}
lora_parameters:
  rank: {config.get('lora_rank', 8)}
  scale: {config.get('lora_scale', 16.0)}
  dropout: {config.get('lora_dropout', 0.05)}
  keys:
{keys_yaml}

# Sequence length
max_seq_length: {config.get('max_seq_length', 2048)}

# Reporting
steps_per_report: 10
steps_per_eval: 100
val_batches: 25
save_every: 100
"""
    LORA_YAML.write_text(yaml)
    print(f"  Updated {LORA_YAML}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_experiments = DEFAULT_EXPERIMENTS
    if "--experiments" in sys.argv:
        idx = sys.argv.index("--experiments")
        n_experiments = int(sys.argv[idx + 1])

    best_config = load_best_config()
    best_bpb = get_best_bpb()

    print("=" * 60)
    print("AUTORESEARCH ORCHESTRATOR")
    print(f"Researcher : {OLLAMA_MODEL} (Ollama, local)")
    print(f"Branch     : {git(['branch', '--show-current']).stdout.strip()}")
    print(f"Experiments: {n_experiments}")
    print(f"Best val_bpb: {best_bpb:.6f}")
    print("=" * 60)
    print()

    for i in range(1, n_experiments + 1):
        print(f"--- Experiment {i}/{n_experiments} ---")

        # 1. Ask researcher
        print("Asking researcher...")
        try:
            response = ask_qwen(read_results(), i, n_experiments, best_config)
            print(f"Researcher: {response[:300]}")
        except Exception as e:
            print(f"Researcher error: {e}, skipping")
            continue

        # 2. Parse
        proposed = parse_config(response)
        if not proposed:
            print("Failed to parse response, skipping")
            continue

        config, description = build_train_config(proposed, best_config)
        print(f"Testing: {description}")

        # 3. Train
        print("Training (5 min)...")
        t0 = time.time()
        try:
            val_bpb, memory_gb, output = run_experiment(config)
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            log_result("timeout", 0.0, 0.0, "crash", f"TIMEOUT: {description}")
            continue
        except Exception as e:
            print(f"CRASH: {e}")
            log_result("crash", 0.0, 0.0, "crash", f"CRASH: {description}")
            continue

        elapsed = time.time() - t0

        if val_bpb is None:
            print("Failed to parse output (crash?)")
            for line in output.strip().split("\n")[-5:]:
                print(f"  {line}")
            log_result("crash", 0.0, 0.0, "crash", f"CRASH: {description}")
            continue

        # 4. Git commit + keep/discard
        improved = val_bpb < best_bpb
        commit_hash = git_commit(description)

        if improved:
            best_bpb = val_bpb
            best_config = save_best_config(config)
            log_result(commit_hash or "unknown", val_bpb, memory_gb, "keep", description)
            print(f"*** NEW BEST: {val_bpb:.6f} *** | mem={memory_gb:.1f}GB | {elapsed:.0f}s | commit {commit_hash}")
        else:
            log_result(commit_hash or "unknown", val_bpb, memory_gb, "discard", description)
            git_reset()
            print(f"val_bpb={val_bpb:.6f} (best={best_bpb:.6f}) | mem={memory_gb:.1f}GB | {elapsed:.0f}s | reverted")

        print()

    # 5. Auto-apply best config to lora_v1.yaml
    print("=" * 60)
    print(f"DONE — {n_experiments} experiments")
    print(f"Best val_bpb: {best_bpb:.6f}")
    print()
    print("Applying best config to finetune/lora_v1.yaml...")
    apply_to_yaml(load_best_config())
    print()
    print("To run full training with optimized hyperparams:")
    print("  cd ~/Projects/ailab1 && bash finetune/train.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
