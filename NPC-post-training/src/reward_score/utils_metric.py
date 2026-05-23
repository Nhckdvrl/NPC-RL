"""Metric configuration per data source for the NPC-RL reward.

The primary metric for a data source is the first entry in its list; its value
is copied to ``score`` and ``acc`` by ``_default_compute_score``. ``format`` is
intentionally absent for npc sources (format is folded into the toolcall F1 /
llm score rather than reported separately).
"""

DATASOURCE_METRICS = {
    "npc/toolcall": ["toolcall_f1"],
    "npc/roleplay": ["llm"],
}

DEFAULT_METRICS = ["score"]


def metrics_for(data_source: str):
    return DATASOURCE_METRICS.get(data_source, DEFAULT_METRICS)
