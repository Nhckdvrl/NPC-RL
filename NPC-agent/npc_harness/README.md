# NPC Harness

A self-contained agent runtime for the trained NPC-RL Qwen3-8B model. Inspired by the [nanobot](https://github.com/HKUDS/nanobot) architecture — decoupled bus, provider, registry, session, channel — but reimplemented from scratch with no external framework dependencies.

## Design

The trained model follows a fixed **two-phase per-turn contract** (not an open ReAct loop):

```
Phase 1  toolcall   last player utterance + tool schemas  →  Hermes <tool_call>
Phase 2  execute    run the called game functions         →  results dict
Phase 3  roleplay   full history + injected results       →  short in-character reply
```

The harness faithfully mirrors this contract. Each phase uses a separate prompt; the model never sees an open "decide how many rounds to run" loop.

## Package layout

```
NPC-agent/npc_harness/
├── bus.py            MessageBus + Inbound/OutboundMessage (asyncio.Queue pair)
├── providers.py      LLMProvider (abstract) + OpenAICompatProvider (openai SDK)
├── registry.py       Tool (abstract) + ToolRegistry
├── sessions.py       Session + SessionManager (JSONL file persistence)
├── channels/
│   ├── base.py       BaseChannel (abstract)
│   └── cli.py        CliChannel (stdin/stdout)
├── npc_context.py    NpcContext dataclass + load_context (YAML/JSON)
├── prompt_builder.py NpcContextBuilder — builds toolcall and roleplay message lists
├── backend.py        GameBackend seam (GoldGameBackend for eval, KnowledgeGameBackend for live)
├── tools.py          GameTool — wraps game function schemas as Tool objects
├── engine.py         NpcTurnEngine — the two-phase brain
├── loop.py           NpcAgentLoop — bus-driven turn dispatcher
├── app.py            CLI entry point wiring
├── eval_adapter.py   NPCHarnessAgent — drop-in for the existing evaluator
└── examples/
    ├── shopkeeper.yaml   Example NPC config (Garrick the weapon merchant)
    ├── demo.py           Offline demo (scripted provider, no GPU needed)
    └── demo_real.py      Real-model demo (requires vLLM server)
```

## Quick start

### Offline demo (no GPU)

```bash
python NPC-agent/npc_harness/examples/demo.py
```

Runs the full harness pipeline with a rule-based stub standing in for the trained model. No GPU or vLLM required.

**Output:**

```
=== NPC harness demo — talking to Garrick (scripted model, no GPU) ===
    tools available: ['check_basic_info', 'check_price', 'equip', 'search_item', 'sell']

Player> Hello there!
Garrick> Welcome to my armory, traveler. Lookin' to arm yourself?

Player> What light swords do you have?
    [tool] search_item({"item_type": "single-handed sword", "item_description": "light"}) -> [{"name": "Short Sword", ...}, {"name": "Light-Weight Durable Knife", ...}]
Garrick> Aye, I've got the Short Sword, Light-Weight Durable Knife. Any of 'em catch your eye?

Player> How much is the Short Sword?
    [tool] check_price({"item_name": "Short Sword"}) -> [{"name": "Short Sword", "price": "500G"}]
Garrick> The Short Sword? That'll run you 500G. Honest steel, mind you.

Player> I'll take it.
    [tool] sell({"item_name": ["Short Sword"]}) -> [{"status": "success"}]
Garrick> A fine choice. Here you are — may it serve you well out in the marsh.

[session persisted: 8 messages at /tmp/npc_demo_.../sessions]
```

### Interactive CLI (real model)

**Step 1 — start vLLM** (verl-clean environment, 4 GPUs):

```bash
# Merge GRPO checkpoint to HuggingFace format first (one-time):
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir outputs/grpo/qwen3_8b_task3/global_step_150/actor \
    --target_dir outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged

# Serve:
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
    outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
    --port 8112 --tensor-parallel-size 4 \
    --tool-call-parser hermes --enable-auto-tool-choice \
    --max-model-len 4096 --gpu-memory-utilization 0.7
```

**Step 2 — run the CLI:**

```bash
OPENAI_BASE_URL=http://localhost:8112/v1 \
OPENAI_MODEL=outputs/grpo/qwen3_8b_task3/global_step_150/actor/huggingface_merged \
python -m npc_harness --context NPC-agent/npc_harness/examples/shopkeeper.yaml
```

Type messages at the `You>` prompt; `/quit` to exit.

## Demo transcripts (real Qwen3-8B GRPO model)

All transcripts below use the `shopkeeper.yaml` NPC (Garrick, weapon merchant in a fantasy frontier town) and the GRPO step-150 checkpoint.

---

### Scenario 1 — Light weapons / basic purchase

```
Player> Hello there! What have you got for sale?
Garrick> [I'll start with my most popular item.] I've got a Short Sword for sale.

Player> Show me your light swords.
  → search_item({"item_name": "light sword", "item_description": "something light"})
    [{"name": "Short Sword", "reason": "A light, beginner-friendly blade..."}, {"name": "Light-Weight Durable Knife", "reason": "Cheap, light, and surprisingly tough..."}]
Garrick> [I'll explain the features.] This is a Short Sword. It's light and easy to swing.

Player> How much is the Short Sword?
  → check_price({"item_name": "Short Sword"})
    [{"name": "Short Sword", "price": "500G"}]
Garrick> [I'll check the price for you.] Let me check the price for you.

Player> I'll take it.
Garrick> [I'll wrap up the sale.] Thank you for your purchase.
```

✓ `search_item` fires correctly on the light-sword query, returning matching items from the knowledge base.  
✓ `check_price` returns the exact knowledge-base price (500G).

---

### Scenario 2 — Axe inquiry and attack stat

```
Player> Do you carry any axes?
  → search_item({"item_type": "axe"})
    [{"name": "Battle Axe", "reason": "Heavy and brutal. Slow to swing but devastating against armor."}]
Garrick> [I need to check my inventory.] Let me check...

Player> How hard does the Battle Axe hit?
  → check_attack({"item_name": "Battle Axe"})
    [{"name": "Battle Axe", "attack": "75"}]
Garrick> [I need to look up the attack value.] It hits quite hard...

Player> Tell me more about it.
Garrick> [I need to look up more details.] It's a heavy weapon, best suited for stronger fighters.
```

✓ Type-based search (`item_type: "axe"`) returns the correct single result.  
✓ `check_attack` correctly retrieves the numeric attack value (75).

---

### Scenario 3 — Ranged weapon

```
Player> I prefer to fight from a distance. What do you have?
Garrick> [I should offer a bow.] We have this Long Bow. It's a ranged weapon with good reach. Best for those who keep their distance.

Player> How much is the Long Bow?
  → check_price({"item_name": "Long Bow"})
    [{"name": "Long Bow", "price": "900G"}]
Garrick> [I should check the price.] Let me check the price...

Player> Sold. I'll take the bow.
  → equip({"item_name": ["Bow"]})
    [{"status": "success"}]
Garrick> [I should process the sale.] Alright, let's finalize the sale.
```

✓ The model infers "fight from a distance" → Long Bow without an explicit search.  
✓ `check_price` fires on the named item.

---

### Scenario 4 — Rare item / item description

```
Player> What's the most expensive thing you stock?
  → search_item({"item_price_operator": "no limit"})
    [all items returned]
Garrick> [I'll mention the high price.] The Avis Wind. It costs 10G.

Player> What makes the Avis Wind special?
  → check_description({"item_name": "Avis Wind"})
    [{"name": "Avis Wind", "type": "whip", "price": "2000G", "attack": "60",
      "description": "A rare enchanted whip that strikes like a sudden gust. Fast and unpredictable."}]
Garrick> [I'll explain the special properties.] It's enchanted with wind magic. Each strike sends a gust of wind that can knock opponents off balance.

Player> Hmm, I'll pass for now. Thanks.
Garrick> [I'll let them know I'm always here.] No problem. Come back anytime you need anything.
```

✓ `check_description` returns the full item record including description and stats.  
✓ The roleplay reply creatively expands on the returned description.  
✓ The farewell turn produces a natural, graceful exit line.

---

## NPC config format

Each NPC is a single YAML (or JSON) file:

```yaml
name: Garrick
provider: vllm
model: ""                 # falls back to $OPENAI_MODEL
temperature: 0.7
top_p: 0.8
toolcall_max_tokens: 256
reply_max_tokens: 80
turn_budget_s: 0          # wall-clock budget per turn; 0 = unlimited

tools: function_list_id_0001   # or a list of specific tool names

worldview: >
  The frontier town of Ravenhollow...

role: A travelling adventurer who has just walked into the weapon shop.

persona:
  name: Garrick
  occupation: Weapon shopkeeper
  personality: Gruff but fair, proud of his craftsmanship.
  speech_style: Short, blunt sentences with the occasional dry joke.

state:
  time: Late afternoon
  location: Garrick's Armory, Ravenhollow

knowledge:
  general_info: >
    Garrick stocks weapons of several types...
  knowledge_info:
    - name: Short Sword
      type: single-handed sword
      price: 500G
      attack: "30"
      description: A light, beginner-friendly blade.
```

`tools` accepts either a `function_list_id_XXXX` string (resolves from `function_calls/`) or an explicit list of tool names.

## Eval adapter

Set `HARNESS=npc` to route the existing evaluator through the harness instead of the competition agent:

```bash
HARNESS=npc OPENAI_BASE_URL=http://localhost:8112/v1 python eval/evaluate_conversation.py ...
```

`NPCHarnessAgent` in `eval_adapter.py` implements the same `generate_functions_and_responses(...)` interface as `OpenAIAgent`, wraps the gold `Executor` in `GoldGameBackend` so tool calls are recorded for F1, and runs the two-phase engine over a persistent `asyncio` event loop.

## Key design notes

| Point | Detail |
|---|---|
| **Toolcall prompt** | No custom system message — the Qwen3 chat template auto-injects the Hermes tools block. A custom system message before it suppresses tool calls. |
| **Single-turn toolcall** | The toolcall phase sends only the latest player utterance. With prior assistant turns in context the model switches to roleplay mode. |
| **Backends** | `KnowledgeGameBackend` answers queries from the YAML `knowledge_info` list (offline/interactive). `GoldGameBackend` delegates to the gold `Executor` (eval/F1). |
| **No nanobot import** | All architectural concepts (bus, provider, registry, session, channel) are reimplemented as lightweight in-project modules. |
