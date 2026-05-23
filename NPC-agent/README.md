# NPC Agent Harness and Service

This folder is the serving layer for the trained NPC-RL model.

- `npc_harness/` contains the core two-phase agent runtime.
- `service/api.py` exposes that runtime through FastAPI for a playable browser demo or game-client prototype.
- The LLM backend is an OpenAI-compatible endpoint, typically vLLM serving the merged SFT/GRPO checkpoint.

## Serve the Model

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir outputs/grpo/qwen3_8b_task3/global_step_150/actor \
  --target_dir outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
  outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
  --port 8112 \
  --tensor-parallel-size 4 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice \
  --max-model-len 4096
```

## Start the FastAPI Service

```bash
OPENAI_BASE_URL=http://localhost:8112/v1 \
OPENAI_MODEL=outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
python3 -m uvicorn --app-dir NPC-agent service.api:app --host 0.0.0.0 --port 8120
```

For a no-GPU smoke test:

```bash
NPC_PROVIDER=scripted python3 -m uvicorn --app-dir NPC-agent service.api:app --host 0.0.0.0 --port 8120
```

The service endpoint is `POST /api/chat`.
