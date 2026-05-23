"""verl custom reward entrypoint.

Point verl at this file:
    custom_reward_function.path=src/verl_reward.py
    custom_reward_function.name=compute_score

verl calls compute_score(data_source, solution_str, ground_truth, extra_info=None)
and reads the "score" key from the returned dict. The actual logic lives in the
reward_score package (toolcall F1 + DeepSeek roleplay judge). Make sure the judge
env vars are exported before launch (see ~/.config/npc-rl/judge.env).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_score import compute_score  # noqa: E402

__all__ = ["compute_score"]
