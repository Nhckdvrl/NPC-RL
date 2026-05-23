"""Real-model demo: talk to the trained Qwen3-8B NPC harness.

Requires a vLLM server already running:

    # start server (verl-clean env, 4 GPUs):
    CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \\
        /home/xiang/NPC-RL/outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \\
        --port 8112 --tensor-parallel-size 4 \\
        --tool-call-parser hermes --enable-auto-tool-choice \\
        --max-model-len 4096

Then run this script:

    OPENAI_BASE_URL=http://localhost:8112/v1 \\
    OPENAI_MODEL=<served-model-name> \\
    python NPC-agent/npc_harness/examples/demo_real.py

Type messages to the NPC, /quit to exit.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from npc_harness.app import build_provider, run_cli


if __name__ == "__main__":
    context = os.path.join(os.path.dirname(__file__), "shopkeeper.yaml")
    asyncio.run(run_cli(context))
