#!/usr/bin/env python3
"""Convert NousResearch/hermes-function-calling-v1 into the project's toolcall
(stage_0) training format.

Emits two aligned artifacts:
  * SFT ShareGPT  -> <out>/sft/stage_0.json
      [{"conversations":[{from,value}...], "tools": "<json-string>"}]
      last turn is a single `function_call` (one {name,arguments} object).
  * verl parquet  -> <out>/verl/stage_0.parquet (+ .json)
      data_source="npc/toolcall", prompt=history, reward_model.ground_truth=
      JSON string of [{name,arguments}], tools=parsed list, extra_info.question.

For a clean first pass we keep only single tool-call assistant turns (exactly one
<tool_call> block), so SFT target and GRPO gold stay consistent. Multi-call
examples can be added later.
"""

import argparse
import json
import os
import re
import glob

import pandas as pd

TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_tool_calls(value: str):
    calls = []
    for inner in TOOLCALL_RE.findall(value or ""):
        try:
            obj = json.loads(inner)
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
    return calls


def tools_to_str(tools) -> str:
    if tools is None:
        return "[]"
    if isinstance(tools, str):
        return tools
    return json.dumps(tools, ensure_ascii=False)


def build_prompt_str(prompt_messages):
    body = "\n".join(
        f"<{m['role']}>{m['content']}</{m['role']}>" for m in prompt_messages
    )
    return "<QUESTION>" + body + "\nWhat tool assistant should call?</QUESTION>"


def role_map(frm: str) -> str:
    return {"human": "user", "gpt": "assistant", "assistant": "assistant",
            "system": "system", "tool": "tool"}.get(frm, frm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="hermes json files (list of {conversations,tools})")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max", type=int, default=0, help="cap #examples (0=all)")
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        files.extend(glob.glob(p))

    sft_rows, verl_rows = [], []
    for fp in files:
        try:
            data = json.load(open(fp))
        except Exception as e:
            print(f"skip {fp}: {e}")
            continue
        for item in data:
            convs = item.get("conversations", [])
            tools_str = tools_to_str(item.get("tools"))
            # find first assistant turn with exactly one tool call
            for idx, turn in enumerate(convs):
                if turn.get("from") in ("gpt", "assistant"):
                    calls = extract_tool_calls(turn.get("value", ""))
                    if len(calls) != 1:
                        continue
                    history = convs[:idx]
                    if not any(t.get("from") == "human" for t in history):
                        continue
                    # SFT shape
                    sft_convs = [{"from": t["from"], "value": t["value"]} for t in history]
                    sft_convs.append({"from": "function_call",
                                      "value": json.dumps(calls[0], ensure_ascii=False)})
                    sft_rows.append({"conversations": sft_convs, "tools": tools_str})
                    # verl shape
                    prompt_messages = [{"role": role_map(t["from"]), "content": t["value"]}
                                       for t in history]
                    verl_rows.append({
                        "id": f"task1_hermes_{len(verl_rows)}",
                        "data_source": "npc/toolcall",
                        "prompt": prompt_messages,
                        "ability": "tool_use",
                        "reward_model": {"ground_truth": json.dumps(calls, ensure_ascii=False),
                                          "style": "rule"},
                        "extra_info": {"task_name": "task1",
                                        "question": build_prompt_str(prompt_messages)},
                        "tools": json.loads(tools_str) if tools_str else [],
                    })
                    break
            if args.max and len(verl_rows) >= args.max:
                break
        if args.max and len(verl_rows) >= args.max:
            break

    os.makedirs(os.path.join(args.out, "sft"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "verl"), exist_ok=True)
    sft_path = os.path.join(args.out, "sft", "stage_0.json")
    json.dump(sft_rows, open(sft_path, "w"), ensure_ascii=False, indent=2)
    verl_json = os.path.join(args.out, "verl", "stage_0.json")
    json.dump(verl_rows, open(verl_json, "w"), ensure_ascii=False, indent=2)
    pd.DataFrame(verl_rows).to_parquet(os.path.join(args.out, "verl", "stage_0.parquet"),
                                       engine="pyarrow", index=False)
    print(f"stage_0: {len(sft_rows)} SFT rows, {len(verl_rows)} verl rows")
    print(f"  -> {sft_path}\n  -> {verl_json} (+.parquet)")


if __name__ == "__main__":
    main()
