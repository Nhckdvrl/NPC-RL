# NPC Post-Training

This folder contains the research pipeline: data construction, SFT cold start, GRPO reinforcement learning, reward modeling, and evaluation.

- `configs/` — LLaMA-Factory, verl, and DeepSpeed configs
- `data/` — dataset registry and raw/processed dataset locations
- `src/` — data transforms, reward functions, synthesis, and analysis
- `eval/` — toolcall and roleplay evaluation
- `function_calls/` — game tool schemas and gold executor (shared with NPC-agent)
- `docs/training_report.md` — full bilingual technical report

---

## Abstract

A two-stage training framework (SFT cold-start → GRPO reinforcement learning) that teaches a single Qwen3-8B model to simultaneously handle game NPC **tool-calling** and **roleplay** dialogue. Starting from a toolcall F1 of 0.32 and roleplay score of 0.52, GRPO training over 150 steps on 4×RTX PRO 6000 GPUs achieves a toolcall F1 of **0.86** and a combined score of **0.69**.

---

## Results

| Stage | Toolcall F1 | Roleplay score | Combined |
|-------|------------:|---------------:|---------:|
| SFT baseline | 0.32 | 0.52 | 0.42 |
| SFT + GRPO (150 steps) | **0.86** | **0.53** | **0.69** |

| Step | Toolcall F1 | Roleplay Score | Combined |
|------|------------:|---------------:|---------:|
| 0 (SFT baseline) | 0.321 | 0.525 | 0.423 |
| 25 | 0.464 | 0.529 | 0.497 |
| 50 | 0.750 | 0.525 | 0.637 |
| 75 | 0.857 | 0.527 | 0.692 |
| 100 | 0.821 | 0.437 | 0.629 |
| 125 | 0.821 | 0.531 | 0.676 |
| **150 (final)** | **0.857** | **0.530** | **0.694** |

---

## Problem Definition

A game NPC agent must handle each conversation turn in two sequential phases:

1. **Toolcall phase**: Given the dialogue history and available game APIs (e.g., `query_inventory`, `get_quest_status`), decide whether a tool call is needed, and if so, generate a correctly formatted `<tool_call>{"name": ..., "parameters": ...}</tool_call>` block.

2. **Roleplay phase**: Given the dialogue history and tool execution results, generate a persona-consistent, contextually appropriate NPC reply (≤200 tokens).

A single model must handle both phases. The evaluation score is the average of toolcall F1 (exact parameter match) and roleplay quality (LLM judge on a 0–1 scale).

---

## Data Pipeline

### Source Datasets

| Dataset | Task | Size | Source |
|---------|------|------|--------|
| Hermes-Function-Calling-v1 | Toolcall (Stage 0) | 928 dialogues | NousResearch / HuggingFace |
| CoSER | Roleplay (Stage 1) | 52,921 dialogues | coser-bench / HuggingFace |

### Dataset Statistics

| Split | npc/toolcall | npc/roleplay | Total |
|-------|-------------|-------------|-------|
| Stage 0 train | 835 | 0 | 835 |
| Stage 1 train | 0 | 47,629 | 47,629 |
| **Full train** | **928 (1.7%)** | **52,921 (98.3%)** | **53,849** |

---

## Stage 1: SFT Cold Start

**Purpose**: Raw Qwen3-8B cannot produce `<tool_call>` formatted outputs or follow the two-phase dialogue pattern. SFT teaches the model the basic output format and task structure before RL begins.

**Framework**: LLaMA-Factory — native ShareGPT format with `function_call` role and `tools` field, out-of-box Qwen3 template support.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen3-8B | Full parameters |
| Fine-tuning type | `full` | No LoRA |
| Template | `qwen3_nothink` | Disables thinking mode |
| DeepSpeed | ZeRO-2 | 4-GPU, gradient checkpointing |
| Learning rate | 1e-5 | Cosine schedule |
| Epochs | 2 | |
| Batch size | 32 effective | per_device=1 × grad_accum=8 × 4 GPUs |
| Max length | 4096 tokens | |

**Thinking mode disabled** (`enable_thinking=False`): NPC replies must stay within 200 tokens; thinking would consume most of the budget. SFT training, GRPO rollout, and inference all use the same no-thinking mode for consistency.

---

## Stage 2: GRPO Reinforcement Learning

**Purpose**: SFT teaches format imitation but cannot optimize for correctness. GRPO directly optimizes toolcall F1 and roleplay quality.

**Algorithm**: Group Relative Policy Optimization samples *n* responses per prompt and uses their relative rewards as the advantage signal — no separate critic network required.

**Framework**: verl — FSDP actor + vLLM rollout colocated on the same GPUs.

| Parameter | Value |
|-----------|-------|
| Group size n | 8 |
| KL penalty | `low_var_kl`, coeff=0.001 |
| Learning rate | 1e-6 |
| Total steps | 150 |
| GPUs | 4×RTX PRO 6000 (~96GB each) |
| Total training time | ~9 hours |

### Reward Functions

**Toolcall reward**: Rule-based F1 of exact function-call matches. Parse all `<tool_call>` blocks, compare against gold by (name, frozenset(parameters)).

**Roleplay reward**: LLM-as-judge via DeepSeek API, 5-dimension rubric (scenario adherence, believability, persona consistency, dialogue flow, functional relevance). Only 1/8 of roleplay rollouts are judged (deterministic MD5 sampling); the rest receive a neutral score of 0.5 — ~87.5% cost reduction.

### Engineering Notes

- **Reward dict key order**: Always initialize `{"toolcall_f1": 0.0, "llm": 0.0}` in fixed order to avoid `DataProto.concat` assertion errors when workers first see different data sources.
- **Torch compile**: Disabled (`use_torch_compile=false`) on RTX PRO 6000 (Blackwell, sm_120) — enabled caused 333s/step vs expected ~210s/step.
- **Checkpoint bucket size**: Increased `update_weights_bucket_megabytes` to 4096 (Qwen3-8B `embed_tokens` is ~2.5GB fp32).
- **CuMemAllocator**: Remove `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — incompatible with verl's allocator.

---

## System Configuration

| Item | Specification |
|------|---------------|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell (96GB each) |
| PyTorch | 2.8 (cu128) |
| transformers | 4.57.6 |
| vLLM | 0.11.0 |
| verl | latest (volcengine/verl) |
| LLaMA-Factory | latest |

---

## Conclusions

1. **GRPO excels at structured output**: Toolcall F1 improved 167% in 150 steps with a simple rule-based reward — no human annotation required.
2. **Roleplay is harder to optimize with RL**: LLM judge scores are noisy and expensive; the SFT cold-start already provides a good roleplay baseline.
3. **Single model, two tasks**: One model handles both toolcalling and roleplay without interference between the two reward types.
4. **Cost**: ~$20–30 in LLM judge API calls with 1/8 sampling; ~9 GPU-hours on 4×96GB GPUs.
