# NPC-RL

A post-training framework for game NPC dialogue models. Trains a single LLM to simultaneously handle **tool-calling** (querying game state) and **roleplay** (generating in-character responses), using SFT cold-start followed by GRPO reinforcement learning.

## Overview

Game NPCs need two capabilities in one turn:
1. **Toolcall** — decide which game APIs to invoke (inventory lookup, quest status, etc.) and format the call correctly
2. **Roleplay** — generate a contextually appropriate, persona-consistent NPC reply after receiving tool results

This repo trains a single Qwen3-8B model to do both, evaluated by toolcall F1 and an LLM judge.

**Training results (Qwen3-8B, 150 GRPO steps):**

| Stage | Toolcall F1 | Roleplay score | Combined |
|-------|------------|----------------|----------|
| SFT baseline | 0.32 | 0.52 | 0.42 |
| After GRPO | **0.86** | **0.53** | **0.69** |

## Project Structure

```
NPC-RL/
├── configs/
│   ├── run_grpo.sh          # GRPO launch script
│   ├── sft_qwen3_8b.yaml    # SFT config (LLaMA-Factory)
│   ├── grpo_qwen3_8b.yaml   # GRPO config (verl)
│   └── ds_z2.json           # DeepSpeed ZeRO-2 config
├── data/
│   ├── dataset_info.json    # LLaMA-Factory dataset registry
│   ├── raw/                 # Raw source data (CoSER + Hermes)
│   ├── sft/                 # SFT training data (ShareGPT format)
│   └── verl/                # GRPO training data (Parquet format)
├── src/
│   ├── reward_score/        # verl reward package (toolcall F1 + LLM judge)
│   ├── data_transform/      # Data conversion scripts
│   ├── dataset_gather/      # Dataset download scripts
│   ├── toolcall_syn/        # Toolcall data synthesis (via LLM API)
│   └── verl_reward.py       # verl reward entrypoint
├── agents/
│   └── openai_agent/        # Inference agent (vLLM + OpenAI-compatible API)
├── eval/                    # Evaluation scripts (toolcall F1 + LLM judge)
└── outputs/                 # Training checkpoints (gitignored)
```

## Setup

### Requirements
- 4× GPU with ≥40GB VRAM (tested on 4×RTX PRO 6000 Blackwell)
- CUDA ≥ 12.8 (Blackwell requires sm_120)
- Python 3.11+

### Environment

```bash
conda create -n npc-rl python=3.11 -y
conda activate npc-rl

# PyTorch — use cu128 for Blackwell, adjust for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install verl          # GRPO trainer
pip install flash-attn    # optional speedup
```

### Base Model

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/qwen3_8b
```

## Data Preparation

### Source Data

| Dataset | Task | Source |
|---------|------|--------|
| [Hermes-Function-Calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Toolcall (Stage 0) | HuggingFace |
| [CoSER](https://huggingface.co/datasets/coser-bench/CoSER) | Roleplay (Stage 1) | HuggingFace |

Download and place under `data/raw/`.

### Build Training Data

```bash
# Stage 0: toolcall (from Hermes)
python src/data_transform/build_stage0_from_hermes.py \
  --input data/raw/hermes_func-calling.json \
  --output data/sft/stage_0.json

# Stage 1: roleplay (from CoSER)
python src/data_transform/build_stage1_from_coser.py \
  --input_dir data/raw/coser/full \
  --output data/sft/stage_1.json

# Merge into stage_all.json for SFT
python src/data_transform/merge_and_tag_data.py

# Convert to Parquet for GRPO
python src/data_transform/convert_to_json_parquet.py \
  --files data/sft/stage_0.json data/sft/stage_1.json \
  --output_dir data/verl
```

## Training

### Stage 1: SFT Cold Start

Uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
# Edit configs/sft_qwen3_8b.yaml to set model_name_or_path
FORCE_TORCHRUN=1 llamafactory-cli train configs/sft_qwen3_8b.yaml
```

Key settings: full-parameter fine-tuning, DeepSpeed ZeRO-2, `qwen3_nothink` template (thinking disabled), lr=1e-5, ~2 epochs.

### Stage 2: GRPO Reinforcement Learning

Uses [verl](https://github.com/volcengine/verl).

```bash
# Toolcall only — rule-based reward, zero API cost (recommended first run)
SFT_CKPT=/path/to/sft/checkpoint bash configs/run_grpo.sh toolcall 150

# Full — toolcall F1 + LLM judge reward (requires judge API key)
SFT_CKPT=/path/to/sft/checkpoint bash configs/run_grpo.sh full 150
```

**LLM Judge setup** (required for `full` mode):

```bash
mkdir -p ~/.config/npc-rl
cat > ~/.config/npc-rl/judge.env << 'EOF'
JUDGE_BASE_URL=https://api.deepseek.com
JUDGE_API_KEY=your_api_key
JUDGE_MODEL=deepseek-chat
JUDGE_TIMEOUT=30
EOF
chmod 600 ~/.config/npc-rl/judge.env
```

Key GRPO settings:
- Algorithm: GRPO, group size n=8
- KL penalty: `low_var_kl`, coef=0.001
- Judge sampling: 1/8 roleplay rollouts (87.5% cost reduction)
- 4-GPU FSDP actor + vLLM rollout (colocate)

## Reward Function

`src/reward_score/` implements the two-source reward:

| Source | Reward | Method |
|--------|--------|--------|
| `npc/toolcall` | F1 of exact tool-call matches | Rule-based |
| `npc/roleplay` | LLM judge score (0–1) | DeepSeek API, 1/8 sampling |

## Inference

```bash
# Serve the trained model via vLLM
python -m vllm.entrypoints.openai.api_server \
  --model outputs/grpo/qwen3_8b_full/global_step_150 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --port 8112

# Use the agent
OPENAI_BASE_URL=http://localhost:8112/v1 python agents/openai_agent/main_agent.py
```
