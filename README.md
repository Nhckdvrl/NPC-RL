# NPC-RL: Post-Training an LLM Game NPC Agent

NPC-RL is a game-NLP research project for post-training a single LLM to act as an interactive NPC: it learns when to call game-state tools, how to format those calls, and how to speak back in character after the game backend returns results.

The main contribution is the post-training pipeline: SFT cold start followed by GRPO reinforcement learning on a combined tool-calling and roleplay objective. The repository also includes an agent harness, vLLM/FastAPI serving path, and a small playable web demo so the trained checkpoint can be shown as an end-to-end prototype.

## Why This Fits Game NLP / LLM R&D

This project is shaped around the kind of work described in an NLP Research Intern role for a game R&D team:

- Research contribution: designs and evaluates a post-training recipe for game NPC dialogue with both tool use and narrative roleplay.
- LLM training: fine-tunes Qwen3-8B with SFT and optimizes it with GRPO using task-specific rewards.
- Prototype to product path: wraps the trained model in a two-phase agent harness and exposes it through vLLM + FastAPI for a browser-playable demo.
- Game-tech relevance: treats game APIs, inventory/quest state, persona grounding, latency budgets, and live backend seams as first-class design constraints.

## Results

Qwen3-8B, trained for 150 GRPO steps after SFT cold start:

| Stage | Toolcall F1 | Roleplay score | Combined |
| --- | ---: | ---: | ---: |
| SFT baseline | 0.32 | 0.52 | 0.42 |
| After GRPO | **0.86** | **0.53** | **0.69** |

## Repository Map

The root keeps the original runnable training paths stable, while the new top-level folders present the project as a complete research-to-demo package. `NPC-post-training/` and `NPC-agent/` use lightweight links back to the original code so old commands and imports keep working.

```text
NPC-RL/
├── NPC-post-training/       # Training pipeline (README has the full technical report)
│   ├── configs/             # SFT, GRPO, DeepSpeed configs
│   ├── data/                # Dataset registry and raw/processed data
│   ├── src/                 # Data transforms, reward functions, synthesis
│   ├── eval/                # Toolcall/roleplay evaluation
│   ├── function_calls/      # Game tool/action schemas and gold executor
│   └── docs/training_report.md  # Bilingual technical report
├── NPC-agent/               # Agent harness + FastAPI serving
│   ├── npc_harness/         # Two-phase NPC runtime (engine, providers, backend, ...)
│   ├── function_calls -> NPC-post-training/function_calls
│   └── service/             # FastAPI wrapper
├── playable-demo/           # Static browser demo for the served NPC
└── outputs/                 # Checkpoints, gitignored
```

## System Design

Each player turn follows the contract used during training:

```text
player utterance + tool schemas
        │
        ▼
Phase 1: toolcall model pass
        │  Hermes/OpenAI tool call JSON
        ▼
Phase 2: game backend execution
        │  inventory / quest / state results
        ▼
Phase 3: roleplay model pass
        │
        ▼
short in-character NPC reply
```

The trained model is not buried in a web UI. It is called in `NPC-agent/npc_harness/engine.py` through `OpenAICompatProvider` in `NPC-agent/npc_harness/providers.py`, which points at a vLLM OpenAI-compatible endpoint.

## Setup

Requirements:

- Python 3.11+
- CUDA-capable GPUs for training; the reported run used 4x RTX PRO 6000 Blackwell
- CUDA 12.8 for Blackwell-class GPUs

```bash
conda create -n npc-rl python=3.11 -y
conda activate npc-rl

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install verl
pip install flash-attn
```

Download the base model:

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir /path/to/qwen3_8b
```

## Data Preparation

| Dataset | Task | Use |
| --- | --- | --- |
| Hermes-Function-Calling-v1 | Tool calling | Cold-start tool-call format and API selection |
| CoSER | Roleplay | Persona-grounded NPC dialogue |

```bash
python3 src/data_transform/build_stage0_from_hermes.py \
  --input data/raw/hermes_func-calling.json \
  --output data/sft/stage_0.json

python3 src/data_transform/build_stage1_from_coser.py \
  --input_dir data/raw/coser/full \
  --output data/sft/stage_1.json

python3 src/data_transform/merge_and_tag_data.py

python3 src/data_transform/convert_to_json_parquet.py \
  --files data/sft/stage_0.json data/sft/stage_1.json \
  --output_dir data/verl
```

## Training

SFT cold start with LLaMA-Factory:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train configs/sft_qwen3_8b.yaml
```

GRPO reinforcement learning with verl:

```bash
SFT_CKPT=/path/to/sft/checkpoint bash configs/run_grpo.sh toolcall 150
SFT_CKPT=/path/to/sft/checkpoint bash configs/run_grpo.sh full 150
```

Reward design:

- `npc/toolcall`: rule-based F1 over exact `(function_name, parameters)` matches.
- `npc/roleplay`: LLM judge score from 0 to 1 over persona consistency, coherence, believability, task relevance, and scenario adherence.
- Judge cost control: deterministic 1/8 roleplay rollout sampling, with neutral reward for unjudged samples.

See [NPC-post-training/docs/training_report.md](NPC-post-training/docs/training_report.md) for the full bilingual technical report.

## Serving the Trained Model

Yes: the trained model can be served directly with vLLM, then wrapped by FastAPI for game/demo traffic.

Merge the verl FSDP checkpoint to HuggingFace format:

```bash
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir outputs/grpo/qwen3_8b_task3/global_step_150/actor \
  --target_dir outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged
```

Serve the merged checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
  outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
  --port 8112 \
  --tensor-parallel-size 4 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --max-model-len 4096
```

Start the NPC API:

```bash
OPENAI_BASE_URL=http://localhost:8112/v1 \
OPENAI_MODEL=outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
python3 -m uvicorn --app-dir NPC-agent service.api:app --host 0.0.0.0 --port 8120
```

No-GPU smoke test:

```bash
NPC_PROVIDER=scripted python3 -m uvicorn --app-dir NPC-agent service.api:app --host 0.0.0.0 --port 8120
```

## Playable Demo

Open [playable-demo/index.html](/home/xiang/NPC-RL/playable-demo/index.html) after starting the FastAPI service, or serve it with `python3 -m http.server 5173 -d playable-demo`. Demo-specific content lives under `playable-demo/`, including `scenarios/ravenhollow_armory.yaml`. The page lets you choose vLLM or scripted smoke-test mode, adjust generation settings, chat with the NPC, and inspect tool traces returned by the agent service.

## Agent Harness

The harness lives in `NPC-agent/npc_harness/` and is deliberately small:

- `engine.py`: two-phase turn engine where the model is invoked.
- `providers.py`: OpenAI-compatible provider for vLLM/SGLang/OpenAI-style APIs.
- `backend.py`: swappable game-state execution seam.
- `tools.py`: converts game function registries into OpenAI tool schemas.
- `sessions.py`: JSONL persistence for chat sessions.
- `eval_adapter.py`: drop-in adapter for evaluator integration.

CLI smoke test without GPU:

```bash
python3 NPC-agent/npc_harness/examples/demo.py
```

Real-model CLI:

```bash
OPENAI_BASE_URL=http://localhost:8112/v1 \
OPENAI_MODEL=outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
python3 -m npc_harness --context NPC-agent/npc_harness/examples/shopkeeper.yaml
```
