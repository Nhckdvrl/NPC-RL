# NPC-RL Post-mortem / 项目复盘

> **EN abstract** — This repository is preserved as-is, together with an honest audit of its failures. NPC-RL trained Qwen3-8B (SFT → GRPO on verl, 4× RTX PRO 6000 Blackwell) for game-NPC tool-calling + roleplay. Toolcall F1 rose 0.32 → 0.86, but the project fails on four levels: (1) data used raw with zero curation and a 1.7%/98.3% task mixture; (2) the roleplay reward filled 7/8 of GRPO rollouts with a constant 0.5, destroying the within-group variance that GRPO's advantage depends on; (3) evaluation had no baselines, no variance/significance, and reused the training judge (Goodhart circularity); (4) the task itself was single-turn format imitation presented as an agentic project — the model never interacted with the environment during RL. The successor project applies the lessons below on a RAGEN-2 (ICML 2026 Oral) recipe with procedurally generated game environments and purely rule-based rewards.
>
> 本文档是对本项目的关键审计，写于项目封存之时（2026-07）。所有问题均附代码位置，可直接核对。

---

## 1. 数据层：零筛选、比例失衡、目标构建幼稚

- 训练数据 = Hermes-Function-Calling 928 段对话（1.7%）+ CoSER 52,921 段对话（98.3%），见 `NPC-post-training/docs/training_report.md` §2.4。CoSER 是整个数据集**原样倒入**：没有质量过滤、没有去重、没有难度分层、没有长度/角色分布分析。
- 数据集**选用与混合本身不成立**：结构化工具调用（可精确验证的 JSON 输出）与文学角色扮演（开放式生成）是目标函数、输出分布、评价方式完全不同的两类任务，没有任何消融或论证支撑"单模型联合训练"这一决定，比例还差出两个数量级——联合训练的设置从立项起就没有正当理由。
- 唯一的"清洗"（`NPC-post-training/src/data_transform/merge_and_tag_data.py`）只校验了 system/human/gpt 的角色交替合法性——这是格式校验，不是数据工程；脚本内还残留 `/path/to/npc-rl/...` 硬编码路径，按提交状态不可复现。
- 训练目标构建方式：从多轮对话中**截取一段工具调用块或一句目标角色台词**作为 target。这既不构成决策监督（不含"何时调用/是否调用"的信号），也让 SFT 与 RL 的监督面严重错位。
- 报告 §2.4 对失衡的辩护（"GRPO 奖励对两类来源独立计算，所以不需要平衡"）是错误推理：奖励独立 ≠ 算力分配合理。98.3% 的 rollout 预算花在了奖励信号已损坏的 roleplay 任务上（见下节），这部分算力近乎全废。

## 2. 奖励层（最致命）：常数填充摧毁 GRPO 组内信号

配置见 `NPC-post-training/configs/run_grpo.sh:49-50`（`JUDGE_SAMPLE_RATE=8`、`JUDGE_SKIP_SCORE=0.5`），实现见 `NPC-post-training/src/reward_score/llm_evaluate.py`。为省 LLM-judge 费用，每条 roleplay rollout 仅以 1/8 概率真实打分，其余直接返回 0.5。三层伤害：

1. **组内方差被抹除**。GRPO 的优势 = 组内奖励的相对差。组大小 8、独立 1/8 抽样下，一个组 8 条全不被抽中的概率为 (7/8)⁸ ≈ **34%**——这三分之一的组奖励全为 0.5，方差为零，advantage 全为 0，纯粹烧卡。约 40% 的组恰好 1 条被判：奖励向量形如 [0.5×7, x]，若 x>0.5，其余 7 条**无论质量好坏**统一获得负 advantage——不是弱信号，是反向噪声。
2. **异常处理把故障变成惩罚**（`llm_evaluate.py:110-111`）：judge 的任何异常（超时、解析失败）返回 **0.0 而非 0.5**——一次网络抖动即把某条 rollout 打成"全组最差"，注入巨大的虚假 advantage。
3. **抽样哈希与内容相关**（`llm_evaluate.py:114-120`）：是否送判由回复**前 64 字节**的 MD5 决定。roleplay 回复常以相同角色名/口头禅开头，导致同组要么全判（成本失控）要么全不判（零方差），进一步退化。

实测后果（`training_report.md` §5.1）：150 步训练中 roleplay 分数 0.525 → 0.530，纹丝不动。**98.3% 的数据 × 约 9 GPU 时的产出为零**；F1 的提升全部来自 1.7% 的规则奖励数据。教训：**奖励的每一分钱成本都必须在算法机制层面预算——省成本的方式是缩数据/换指标，绝不是稀释信号。**

## 3. 评测层：无基线、无统计、自己给自己打分

- 全部结果只有 SFT vs GRPO 两行：没有 base 模型 zero-shot、没有任何外部模型对照、没有 RL-only 消融。
- "Combined score" 把规则 F1 与 LLM-judge 分**直接平均**——两个量纲、噪声结构、效度状态完全不同的量，合成数无测量学意义。
- 评测 judge 与训练 reward 是**同一个 DeepSeek 模型 + 同一套 rubric**：RL 直接优化的就是这个打分函数，终评用它属于循环评估（Goodhart；参见 Gao et al., ICML 2023, arXiv:2210.10760——proxy 分数上涨的同时真实质量可以下降）。
- 无方差、无置信区间、无显著性检验。事后计算：0.525→0.530（0.5pp）的差异，配对检验也需要约 3 万条评测样本才能建立——在本项目任何现实评测规模下**这个数字不可证伪，等于把噪声当成果汇报**。
- `trainer.logger=['console']`（`run_grpo.sh:109`）——没有开 wandb，报告中的熵/KL 曲线没有任何可出示的原始记录。

## 4. 任务设计层（根因）：单轮格式模仿伪装成 agentic 项目

- 训练目标"输出 Hermes 格式 `<tool_call>`"是 2025 年后所有主流模型的出厂能力。这决定了本项目只能选未做过 agentic 优化的模型来"训出提升"——方法不可迁移，被面试官一针见血指出。
- 更深一层：**本项目根本不是 agentic RL**。verl 训练数据是单轮 prompt → response 对静态 gold 打分（`max_response_length=256`），模型在整个 RL 过程中从未与游戏后端交互过一步；"两阶段 agent 循环"只存在于推理侧 demo（`NPC-agent/`）。何时调用工具、如何依据环境反馈决定下一步（继续推理/行动/终止）——这些真正的 agentic 能力全靠推理侧 harness 打补丁，训练对此毫无贡献。
- 正确的立项方式恰恰相反：工具调用**本身**应该作为多轮环境交互 RL 的内容——让模型在环境反馈闭环里学会"何时调用、调用后如何继续"，奖励来自环境终态。这是后继项目的出发点。

## 5. 有效资产（本项目仍然成立的部分）

1. **规则奖励 + GRPO 的那一小片是健康的**：toolcall F1 0.32 → 0.86（150 步），证明可验证奖励下 GRPO 对结构化输出确实有效——这也反衬了 roleplay 侧的失败是奖励设计问题，不是算法问题。
2. **sm_120（RTX PRO 6000 Blackwell）上被验证可用的训练栈**：torch 2.8 cu128 + vLLM 0.11.0 + transformers 4.57.6 + flash-attn 2.8.3 + verl，GRPO ~210s/步。截至封存时，公开渠道找不到第二个在此卡上跑通 verl RL 的记录。
3. **五个真实排障记录**（`training_report.md` §4.6）：
   - Blackwell 上 `torch.compile` 不稳定（333s/步 → 关闭后 210s/步）；
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 verl `CuMemAllocator` 冲突；
   - Qwen3-8B `embed_tokens` fp32 ~2.5GB 超出默认 checkpoint bucket（2048MB → 4096MB）；
   - transformers 5.x → 4.57.6 降级后 `tokenizer_config.json` 的 `extra_special_tokens` 格式不兼容；
   - 多 worker 下 reward dict 插入顺序不一致导致 `DataProto.concat` AssertionError（固定 key 序修复）。

---

*Audited and written at archive time, 2026-07. All file references are to this repository's final state.*
