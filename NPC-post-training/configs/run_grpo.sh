#!/usr/bin/env bash
# Launch verl GRPO training for NPC-RL.
# Usage:
#   bash configs/run_grpo.sh toolcall      # Stage 0 only — toolcall F1 reward, zero API cost (default)
#   bash configs/run_grpo.sh full          # Stage 0+1 — toolcall F1 + LLM judge reward
#   bash configs/run_grpo.sh full 200      # full mode, 200 steps

set -euo pipefail

MODE="${1:-toolcall}"
STEPS="${2:-150}"

# ── Judge env (only needed for full mode) ────────────────────────────────────
JUDGE_ENV="$HOME/.config/npc-rl/judge.env"
if [[ "$MODE" == "full" ]] && [[ -f "$JUDGE_ENV" ]]; then
    set -a && . "$JUDGE_ENV" && set +a
    echo "[run_grpo] Loaded judge env: model=${JUDGE_MODEL:-unset}"
fi

# ── Python ────────────────────────────────────────────────────────────────────
PYTHON="${PYTHON:-$(which python)}"

# ── Data / output ─────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_CKPT="${SFT_CKPT:-${REPO}/outputs/sft/qwen3_8b/checkpoint-final}"

if [[ "$MODE" == "toolcall" ]]; then
    TRAIN_FILES="['${REPO}/data/verl/stage_0_train.parquet']"
    VAL_FILES="['${REPO}/data/verl/stage_0_val.parquet']"
    OUT_DIR="${REPO}/outputs/grpo/qwen3_8b_toolcall"
    EXP_NAME="grpo-toolcall-qwen3-8b"
elif [[ "$MODE" == "full" ]]; then
    # full_train.parquet = stage_0 (toolcall) + stage_1 (roleplay)
    TRAIN_FILES="['${REPO}/data/verl/full_train.parquet']"
    VAL_FILES="['${REPO}/data/verl/full_val.parquet']"
    OUT_DIR="${REPO}/outputs/grpo/qwen3_8b_full"
    EXP_NAME="grpo-full-qwen3-8b"
else
    echo "Unknown mode: $MODE. Use 'toolcall' or 'full'." && exit 1
fi

mkdir -p "$OUT_DIR"

# ── Env vars ──────────────────────────────────────────────────────────────────
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/npc_rl_triton_grpo"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# judge: only call API for 1/8 of roleplay rollouts → ~87.5% cost reduction
export JUDGE_SAMPLE_RATE=8
export JUDGE_SKIP_SCORE=0.5

mkdir -p "$TRITON_CACHE_DIR"

echo "====================================================="
echo " NPC-RL GRPO | mode=$MODE steps=$STEPS"
echo " train: $TRAIN_FILES"
echo " out:   $OUT_DIR"
echo "====================================================="

cd "$REPO"

# verl's main_ppo uses its own config dir; we pass all overrides on the CLI
$PYTHON -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=256 \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=true \
    data.truncation=left \
    "+data.apply_chat_template_kwargs={enable_thinking: false}" \
    actor_rollout_ref.model.path="${SFT_CKPT}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.max_model_len=2560 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=true \
    algorithm.use_kl_in_reward=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward.custom_reward_function.path="${REPO}/src/verl_reward.py" \
    reward.custom_reward_function.name=compute_score \
    reward.num_workers=4 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$STEPS" \
    trainer.project_name=npc-rl \
    trainer.experiment_name="$EXP_NAME" \
    trainer.logger="['console']" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.default_local_dir="$OUT_DIR" \
    trainer.default_hdfs_dir=null
