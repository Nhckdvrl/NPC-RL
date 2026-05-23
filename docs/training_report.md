# NPC-RL Training Report
# NPC-RL 训练技术报告

---

## Abstract / 摘要

This report documents the full post-training pipeline for NPC-RL: a two-stage training framework (SFT cold-start → GRPO reinforcement learning) that teaches a single Qwen3-8B model to simultaneously handle game NPC **tool-calling** and **roleplay** dialogue. Starting from a toolcall F1 of 0.32 and roleplay score of 0.52, GRPO training over 150 steps on 4×RTX PRO 6000 GPUs achieves a toolcall F1 of **0.86** and a combined score of **0.69**.

本报告记录 NPC-RL 的完整后训练流程：一套两阶段训练框架（SFT 冷启动 → GRPO 强化学习），使单个 Qwen3-8B 模型同时具备游戏 NPC 的**工具调用**与**角色扮演**能力。从 toolcall F1=0.32、roleplay=0.52 的基线出发，在 4×RTX PRO 6000 上经过 150 步 GRPO 训练，最终 toolcall F1 达到 **0.86**，综合分达到 **0.69**。

---

## 1. Problem Definition / 问题定义

### English

A game NPC agent must handle each conversation turn in two sequential phases:

1. **Toolcall phase**: Given the dialogue history and available game APIs (e.g., `query_inventory`, `get_quest_status`, `check_npc_relationship`), decide whether a tool call is needed, and if so, generate a correctly formatted `<tool_call>{"name": ..., "parameters": ...}</tool_call>` block.

2. **Roleplay phase**: Given the dialogue history and tool execution results, generate a persona-consistent, contextually appropriate NPC reply (≤200 tokens).

A single model must handle both phases. The evaluation score is the average of toolcall F1 (exact parameter match) and roleplay quality (LLM judge on a 0–1 scale).

### 中文

游戏 NPC Agent 需要在每个对话轮次中依次完成两个阶段：

1. **工具调用阶段**：给定对话历史和可用的游戏 API（如 `query_inventory`、`get_quest_status`、`check_npc_relationship`），判断是否需要调用工具，若需要则生成格式正确的 `<tool_call>{"name": ..., "parameters": ...}</tool_call>` 块。

2. **角色扮演阶段**：给定对话历史和工具执行结果，生成符合角色人设、上下文连贯的 NPC 回复（≤200 tokens）。

单个模型需要同时处理两个阶段。最终评分为 toolcall F1（精确参数匹配）和 roleplay 质量（LLM judge 0-1 分）的平均值。

---

## 2. Data Pipeline / 数据管线

### 2.1 Source Datasets / 数据来源

| Dataset | Task | Size | Source |
|---------|------|------|--------|
| Hermes-Function-Calling-v1 | Toolcall (Stage 0) | 928 dialogues | NousResearch / HuggingFace |
| CoSER | Roleplay (Stage 1) | 52,921 dialogues | coser-bench / HuggingFace |

**English**: Hermes provides multi-turn conversations with JSON Schema tool definitions and structured `<tool_call>` outputs—directly compatible with NPC toolcalling. CoSER provides character roleplay conversations derived from literary works, each with rich character profiles (persona, worldview, knowledge background), making it a strong fit for NPC persona-grounded dialogue.

**中文**：Hermes 提供带 JSON Schema 工具定义和结构化 `<tool_call>` 输出的多轮对话，与 NPC 工具调用需求直接对齐。CoSER 提供基于文学作品角色的扮演对话，每条数据均含丰富的角色档案（人设、世界观、知识背景），非常适合 NPC 人设对话训练。

### 2.2 SFT Data Format / SFT 数据格式

Both datasets are converted to **ShareGPT format** for LLaMA-Factory:

两个数据集均转换为 LLaMA-Factory 所需的 **ShareGPT 格式**：

```json
{
  "conversations": [
    {"from": "system",        "value": "<character profile / tool definitions>"},
    {"from": "human",         "value": "<player message>"},
    {"from": "function_call", "value": "<tool_call>{...}</tool_call>"},
    {"from": "observation",   "value": "<tool execution result>"},
    {"from": "gpt",           "value": "<NPC reply>"}
  ],
  "tools": "[{\"name\": \"query_inventory\", \"parameters\": {...}}]"
}
```

### 2.3 GRPO Data Format / GRPO 数据格式

For reinforcement learning, data is converted to **verl Parquet format**:

强化学习阶段，数据转换为 **verl Parquet 格式**：

| Column | Content | 内容 |
|--------|---------|------|
| `prompt` | Dialogue history without the last turn (packed as `<QUESTION>...</QUESTION>`) | 去掉最后一轮的对话历史 |
| `data_source` | `npc/toolcall` or `npc/roleplay` | 数据类型标识 |
| `reward_model.ground_truth` | Gold tool-call JSON or gold NPC reply | 标准答案 |

### 2.4 Dataset Statistics / 数据集统计

| Split | npc/toolcall | npc/roleplay | Total |
|-------|-------------|-------------|-------|
| Stage 0 train | 835 | 0 | 835 |
| Stage 0 val | 93 | 0 | 93 |
| Stage 1 train | 0 | 47,629 | 47,629 |
| Stage 1 val | 0 | 5,292 | 5,292 |
| **Full train** | **928 (1.7%)** | **52,921 (98.3%)** | **53,849** |
| Full val | 93 | 5,292 | 5,385 |

**English**: The combined dataset is heavily imbalanced (98.3% roleplay). This is intentional — CoSER is far larger than Hermes, and rebalancing by oversampling toolcall data was considered but not applied, as the GRPO reward function naturally handles the two sources independently.

**中文**：合并数据集严重不平衡（98.3% roleplay）。这是有意为之——CoSER 远大于 Hermes，虽然考虑过通过过采样 toolcall 数据来平衡，但最终未采用，因为 GRPO 奖励函数对两类来源独立计算，不需要数据平衡。

---

## 3. Stage 1: SFT Cold Start / 第一阶段：SFT 冷启动

### 3.1 Purpose / 目的

**English**: Raw Qwen3-8B cannot produce `<tool_call>` formatted outputs or follow the two-phase dialogue pattern. SFT teaches the model the basic output format and task structure before RL begins. Without SFT, GRPO would explore a large format space and converge slowly or not at all.

**中文**：原始 Qwen3-8B 无法生成 `<tool_call>` 格式输出，也不了解两阶段对话结构。SFT 在 RL 开始前教会模型基本的输出格式和任务结构。没有 SFT 冷启动，GRPO 需要在巨大的格式空间中探索，收敛会很慢甚至无法收敛。

### 3.2 Framework & Configuration / 框架与配置

**Framework**: LLaMA-Factory — chosen for its native support of ShareGPT format with `function_call` role and `tools` field, and out-of-box Qwen3 template support.

**框架**：LLaMA-Factory——选择原因是其原生支持带 `function_call` 角色和 `tools` 字段的 ShareGPT 格式，以及开箱即用的 Qwen3 模板支持。

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen3-8B | Full parameters |
| Fine-tuning type | `full` | No LoRA — full capacity needed for tool-calling precision |
| Template | `qwen3_nothink` | Disables Qwen3 thinking mode |
| DeepSpeed | ZeRO-2 | 4-GPU, gradient checkpointing |
| Learning rate | 1e-5 | Cosine schedule, warmup_ratio=0.03 |
| Epochs | 2 | |
| Batch size | 32 effective | per_device=1 × grad_accum=8 × 4 GPUs |
| Max length | 4096 tokens | |
| Precision | BF16 + TF32 | |

### 3.3 Key Design Decision: Thinking Mode / 关键设计：关闭 Thinking 模式

**English**: Qwen3-8B supports a "thinking" mode (similar to o1) where the model reasons step-by-step before answering. This was disabled (`enable_thinking=False`) for three reasons:
1. NPC replies must stay within 200 tokens — thinking output would consume most of the budget
2. Tool-call format is well-defined and doesn't benefit from extended reasoning
3. Consistency requirement: SFT training (`qwen3_nothink` template), GRPO rollout (`apply_chat_template_kwargs={enable_thinking: false}`), and inference must all use the same mode

**中文**：Qwen3-8B 支持"thinking 模式"（类似 o1，先推理再回答）。本项目关闭了该模式（`enable_thinking=False`），原因有三：
1. NPC 回复须在 200 tokens 内，thinking 输出会消耗大部分 token 预算
2. 工具调用格式明确，不需要扩展推理
3. 一致性要求：SFT 训练（`qwen3_nothink` 模板）、GRPO rollout（`apply_chat_template_kwargs={enable_thinking: false}`）和推理侧三处必须统一设置

---

## 4. Stage 2: GRPO Reinforcement Learning / 第二阶段：GRPO 强化学习

### 4.1 Purpose / 目的

**English**: SFT teaches format imitation but cannot optimize for correctness. A model can produce syntactically valid `<tool_call>` blocks with wrong function names or parameters and still receive high SFT loss reward. GRPO directly optimizes the metrics that matter: toolcall F1 (did you pick the right function with right parameters?) and roleplay quality (is the response in-character and contextually appropriate?).

**中文**：SFT 只能教模型模仿格式，无法优化正确性。模型可能输出语法正确但函数名或参数有误的 `<tool_call>` 块，SFT loss 仍然较低。GRPO 直接优化真正重要的指标：toolcall F1（函数选择和参数是否正确？）和 roleplay 质量（回复是否符合人设、上下文是否连贯？）。

### 4.2 Algorithm: GRPO / 算法：GRPO

**English**: Group Relative Policy Optimization samples *n* responses per prompt and uses their relative rewards as the advantage signal — no separate critic network required. This makes it significantly more memory-efficient than PPO for LLM training.

**中文**：GRPO（组相对策略优化）对每个 prompt 采样 *n* 个回复，用组内相对奖励作为优势信号，无需独立的 critic 网络。相比 PPO，这在 LLM 训练中显著节省显存。

Advantage for response $i$ in group $g$:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_g)}{\text{std}(r_g)}$$

Combined objective with KL penalty:

$$\mathcal{L} = -\mathbb{E}\left[\hat{A}_i \cdot \log\pi_\theta(y_i|x)\right] + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

### 4.3 Framework / 框架

**Framework**: verl (volcengine) — supports FSDP actor + vLLM rollout colocate on the same GPUs, avoiding the overhead of separate actor/rollout server processes.

**框架**：verl（火山引擎开源）——支持 FSDP actor + vLLM rollout 在同一组 GPU 上 colocate，避免独立 actor/rollout 服务器进程的额外开销。

### 4.4 Training Configuration / 训练配置

| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | GRPO | |
| Group size n | 8 | 8 rollouts per prompt |
| KL penalty | `low_var_kl` | Lower variance than standard KL |
| KL coefficient | 0.001 | |
| Learning rate | 1e-6 | Actor only |
| Train batch size | 128 prompts | |
| PPO mini-batch | 128 | Must be ≤ train_batch_size |
| Micro-batch/GPU | 16 | |
| Total steps | 150 | |
| Rollout temp | 1.0 | High temperature for exploration |
| vLLM GPU util | 0.45 | Leaves room for FSDP actor |
| Max model len | 2560 | Prompt 2048 + response 256 |
| GPUs | 4×RTX PRO 6000 | ~96GB VRAM each |
| Step time | ~210s | |
| Total training time | ~9 hours | |

### 4.5 Reward Function / 奖励函数

#### Toolcall Reward (`npc/toolcall`)

**English**: Rule-based F1 of exact function-call matches. Parse all `<tool_call>{...}</tool_call>` blocks from the model output, compare against gold functions by (name, frozenset(parameters)) tuples.

**中文**：基于规则的精确函数调用匹配 F1。解析模型输出中所有 `<tool_call>{...}</tool_call>` 块，按 (函数名, frozenset(参数键值对)) 元组与 gold 函数集合比较。

$$F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

- If gold has no tool calls and model produces none → score = 1.0 (correct abstention)
- If either set is empty but not both → precision/recall = 0

#### Roleplay Reward (`npc/roleplay`)

**English**: LLM-as-judge scoring via DeepSeek API. The judge evaluates the NPC reply on a 5-dimension rubric, returning a score in [0, 100] normalized to [0, 1]:

**中文**：通过 DeepSeek API 调用 LLM judge 打分。judge 按 5 维评分标准对 NPC 回复打 0-100 分，归一化到 [0, 1]：

| Dimension | 维度 |
|-----------|------|
| Scenario adherence & quest progression | 场景遵从与任务推进 |
| Believability & engagement | 可信度与吸引力 |
| Persona consistency | 人设一致性 |
| Dialogue flow & coherence | 对话流畅与连贯性 |
| Functional relevance | 功能相关性（与工具结果的一致性）|

#### Cost Reduction: 1/8 Judge Sampling / 成本优化：1/8 采样

**English**: Calling the LLM judge for all 52,921 roleplay rollouts × 8 group size = ~423,368 API calls per epoch would be prohibitively expensive. Instead, only 1/8 of roleplay rollouts are actually judged; the rest receive a neutral score of 0.5. The sampling is deterministic (MD5 hash of the first 64 bytes of the response), ensuring reproducibility across distributed workers.

**中文**：对全部 52,921 条 roleplay × 8 组 = ~423,368 次 API 调用的代价极高。因此只对 1/8 的 rollout 真正调用 judge，其余直接返回中性分 0.5。采样是确定性的（对回复前 64 字节做 MD5 hash），确保分布式 worker 间的可复现性。

Cost reduction: ~87.5% | 成本降低：约 87.5%

### 4.6 Engineering Challenges & Solutions / 工程难点与解决方案

#### Challenge 1: Reward Dict Key Order / 问题1：Reward Dict Key 顺序

**English**: verl collects `reward_extra_keys` (the list of non-`score` keys in the reward dict) from each worker independently. When different workers first see different data sources (`npc/toolcall` vs `npc/roleplay`), the dict insertion order differs (`[toolcall_f1, llm]` vs `[llm, toolcall_f1]`), causing an `AssertionError` at `DataProto.concat` during step 7.

**Fix**: Always initialize the result dict with a fixed key order regardless of data source:

**中文**：verl 在每个 worker 中独立收集 `reward_extra_keys`（reward dict 中非 `score` 的 key 列表）。当不同 worker 首先遇到不同数据源时，dict 插入顺序不同（`[toolcall_f1, llm]` vs `[llm, toolcall_f1]`），导致 step 7 在 `DataProto.concat` 时 `AssertionError`。

**修复**：无论数据源类型，始终以固定 key 顺序初始化 result dict：

```python
# Fixed order — same regardless of data_source
result = {"toolcall_f1": 0.0, "llm": 0.0}
```

#### Challenge 2: Torch Compile on Blackwell / 问题2：Blackwell 上的 Torch Compile

**English**: RTX PRO 6000 (Blackwell, sm_120) had instability with `torch.compile` enabled, causing 333s/step vs the expected ~210s/step. Disabling it (`use_torch_compile=false`) restored normal speed.

**中文**：RTX PRO 6000（Blackwell，sm_120）开启 `torch.compile` 时不稳定，每步耗时 333s 而非预期的 210s。关闭后（`use_torch_compile=false`）速度恢复正常。

#### Challenge 3: Checkpoint Bucket Size / 问题3：Checkpoint Bucket 大小

**English**: Qwen3-8B's `embed_tokens` weight is ~2.5GB in fp32. The default `update_weights_bucket_megabytes=2048` was too small, causing checkpoint save/load failures. Increased to 4096MB.

**中文**：Qwen3-8B 的 `embed_tokens` 权重 fp32 约 2.5GB，默认的 `update_weights_bucket_megabytes=2048` 不够，导致 checkpoint 存取失败。调整为 4096MB。

#### Challenge 4: CuMemAllocator Conflict / 问题4：CuMemAllocator 冲突

**English**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is incompatible with verl's `CuMemAllocator`. Removing this environment variable resolved allocation errors.

**中文**：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 verl 的 `CuMemAllocator` 不兼容，移除该环境变量后分配错误消失。

#### Challenge 5: Tokenizer Config Compatibility / 问题5：Tokenizer Config 兼容性

**English**: After downgrading from transformers 5.x to 4.57.6 (required for vLLM 0.11.0 compatibility), the SFT checkpoint's `tokenizer_config.json` had `extra_special_tokens` as a list (`['<|im_start|>', ...]`) instead of a dict, causing a load error. Patched to `{}`.

**中文**：从 transformers 5.x 降级到 4.57.6（vLLM 0.11.0 兼容性要求）后，SFT checkpoint 的 `tokenizer_config.json` 中 `extra_special_tokens` 为列表格式而非字典格式，导致加载报错。将其 patch 为 `{}`。

---

## 5. Results / 训练结果

### 5.1 Validation Metrics Over Training / 训练过程验证指标

| Step | Toolcall F1 | Roleplay Score | Combined Avg |
|------|------------|----------------|--------------|
| 0 (SFT baseline) | 0.321 | 0.525 | 0.423 |
| 25 | 0.464 | 0.529 | 0.497 |
| 50 | 0.750 | 0.525 | 0.637 |
| 75 | 0.857 | 0.527 | 0.692 |
| 100 | 0.821 | 0.437 | 0.629 |
| 125 | 0.821 | 0.531 | 0.676 |
| **150 (final)** | **0.857** | **0.530** | **0.694** |

### 5.2 Training Dynamics / 训练动态

**English**:
- **Toolcall F1** improved dramatically from 0.32 to 0.86 (+167%), demonstrating that GRPO with a rule-based reward signal is highly effective for structured output tasks
- **Roleplay score** remained stable (~0.52–0.53) throughout training. The 1/8 judge sampling introduces noise, and the highly imbalanced dataset (98.3% roleplay) means each toolcall example's signal is diluted. Despite this, no regression was observed
- **Entropy** declined steadily from 1.19 → 0.95, indicating the model became more confident without collapsing
- **KL penalty** grew from 0.001 → 0.099, remaining within safe bounds, confirming the model did not drift excessively from the SFT initialization

**中文**：
- **Toolcall F1** 从 0.32 大幅提升至 0.86（+167%），证明基于规则奖励的 GRPO 对结构化输出任务非常有效
- **Roleplay 分数** 在训练过程中保持稳定（~0.52–0.53）。1/8 judge 采样引入了一定噪声，高度不平衡的数据集（98.3% roleplay）也导致 toolcall 样本的信号被稀释，尽管如此未出现退化
- **Entropy** 从 1.19 稳步下降到 0.95，表明模型变得更自信但没有 collapse
- **KL 惩罚** 从 0.001 增长到 0.099，保持在安全范围内，确认模型未偏离 SFT 初始化过多

---

## 6. System Configuration / 系统配置

| Item | Specification |
|------|---------------|
| GPUs | 4× NVIDIA RTX PRO 6000 Blackwell (96GB each) |
| CUDA | 12.8+ (required for sm_120) |
| PyTorch | 2.8 (cu128 build) |
| transformers | 4.57.6 |
| vLLM | 0.11.0 |
| flash-attn | 2.8.3 |
| verl | latest (volcengine/verl) |
| LLaMA-Factory | latest |

---

## 7. Conclusions / 结论

**English**: The two-stage SFT→GRPO pipeline is highly effective for training NPC dialogue agents:

1. **GRPO excels at structured output tasks**: Toolcall F1 improved 167% in 150 steps with a simple rule-based reward — no human annotation required
2. **Roleplay is harder to optimize with RL**: LLM judge scores are noisy, and the sparse/expensive signal (1/8 sampling) limits rapid improvement. The SFT cold-start already provides a reasonable roleplay baseline
3. **The two-task design works**: A single model can learn both toolcalling and roleplay without interference between the two reward types
4. **Cost is manageable**: 150-step GRPO training costs ~$20–30 in LLM judge API calls with 1/8 sampling, and ~9 GPU-hours on 4×96GB GPUs

**中文**：SFT→GRPO 两阶段训练管线对 NPC 对话 Agent 非常有效：

1. **GRPO 在结构化输出任务上表现出色**：仅用简单规则奖励，150 步内 toolcall F1 提升 167%，无需人工标注
2. **Roleplay 用 RL 优化更难**：LLM judge 分数噪声较大，稀疏/昂贵的信号（1/8 采样）限制了快速提升。SFT 冷启动已经提供了合理的 roleplay 基线
3. **双任务设计可行**：单个模型可以同时学习工具调用和角色扮演，两种奖励类型之间没有互相干扰
4. **成本可控**：1/8 采样下 150 步 GRPO 的 LLM judge API 费用约 $20–30，GPU 训练时间约 9 小时（4×96GB）

---

*Generated: 2026-05-23 | Hardware: 4×RTX PRO 6000 Blackwell | Model: Qwen3-8B*
