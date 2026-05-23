# NPC Post-Training

This is the research-heavy part of the project: data construction, SFT cold start, GRPO reinforcement learning, reward modeling, and evaluation.

The active code remains in the repository root to keep existing training commands stable:

- `../configs/` - LLaMA-Factory, verl, and DeepSpeed configs
- `../data/` - dataset registry and raw/processed dataset locations
- `../src/` - data transforms, reward functions, synthesis, and analysis
- `../eval/` - toolcall and roleplay evaluation
- `../docs/training_report.md` - bilingual technical report

## Research Focus

The model is post-trained to combine two game-NPC skills in a single Qwen3-8B policy:

1. Tool calling: select and format game API calls with exact parameters.
2. Roleplay: produce short, grounded, persona-consistent NPC dialogue after tool execution.

The pipeline uses SFT for format/task cold start and GRPO to optimize the evaluation signal directly. Tool calls receive rule-based F1 reward; roleplay receives an LLM-judge reward with deterministic subsampling for cost control.

## Core Result

| Stage | Toolcall F1 | Roleplay score | Combined |
| --- | ---: | ---: | ---: |
| SFT baseline | 0.32 | 0.52 | 0.42 |
| SFT + GRPO | 0.86 | 0.53 | 0.69 |
