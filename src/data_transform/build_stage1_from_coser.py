#!/usr/bin/env python3
"""Convert Neph0s/CoSER (per-book json) into the project's roleplay (stage_1)
training format.

Each CoSER plot conversation has: scenario, topic, key_characters[{name,thought}],
dialogues[{character,message}]. We pick each well-profiled target character and
build an in-character dialogue ending on that character's line (the gold reply).

Emits:
  * SFT ShareGPT  -> <out>/sft/stage_1.json   [{"conversations":[{from,value}...]}]
       turns: system (persona+scene), then human/gpt alternation, ending in gpt.
  * verl parquet  -> <out>/verl/stage_1.parquet (+ .json)
       data_source="npc/roleplay", prompt=history, reward_model.ground_truth=
       gold reply, extra_info.question=full history (so the DeepSeek judge has
       the scene + dialogue context, not just the gold line).
"""

import argparse
import glob
import json
import os

import pandas as pd

MAX_CONTEXT_TURNS = 10  # dialogue turns kept before the gold reply


def make_system(name, profile, scenario, topic, thought):
    parts = [f"Now you play as {name} in a roleplay scene. Stay fully in character."]
    if profile:
        parts.append(f"# Character Profile\n{profile}")
    if scenario:
        parts.append(f"# Scene\n{scenario}")
    if topic:
        parts.append(f"# Topic\n{topic}")
    if thought:
        parts.append(f"# Your current inner motivation\n{thought}")
    parts.append("## Instructions\n- Respond in character as "
                 f"{name}.\n- Be consistent with the profile and scene.\n"
                 f"- Reply with only {name}'s next line.")
    return "\n\n".join(parts)


def build_prompt_str(prompt_messages):
    body = "\n".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in prompt_messages)
    return "<QUESTION>" + body + "\nWhat assistant should response?</QUESTION>"


def role_map(frm):
    return {"human": "user", "gpt": "assistant", "system": "system"}.get(frm, frm)


def build_examples_for_conv(conv, profiles, max_per_conv):
    scenario = conv.get("scenario", "")
    topic = conv.get("topic", "")
    dialogues = conv.get("dialogues", []) or []
    thoughts = {kc.get("name"): kc.get("thought", "")
                for kc in (conv.get("key_characters") or []) if isinstance(kc, dict)}
    speakers = [d for d in dialogues if isinstance(d, dict) and d.get("message")]
    targets = [c for c in {d["character"] for d in speakers}
               if c in profiles and any(d["character"] == c for d in speakers)]
    out = []
    for name in targets:
        # walk dialogues, mapping to human/gpt; cut at the last target line
        last_target = max((i for i, d in enumerate(speakers) if d["character"] == name),
                          default=-1)
        if last_target < 0:
            continue
        seq = speakers[: last_target + 1]
        if len(seq) > MAX_CONTEXT_TURNS:
            seq = seq[-MAX_CONTEXT_TURNS:]
            if seq[-1]["character"] != name:  # safety
                continue
        msgs = [{"from": "system",
                 "value": make_system(name, profiles.get(name, ""), scenario, topic,
                                       thoughts.get(name, ""))}]
        for d in seq:
            if d["character"] == name:
                msgs.append({"from": "gpt", "value": d["message"].strip()})
            else:
                msgs.append({"from": "human",
                             "value": f"{d['character']}: {d['message'].strip()}"})
        # ensure a human turn precedes the first gpt
        if msgs[1]["from"] == "gpt":
            msgs.insert(1, {"from": "human",
                            "value": f"[The scene begins.] {topic or scenario[:200]}"})
        if msgs[-1]["from"] != "gpt":
            continue
        if not any(m["from"] == "human" for m in msgs):
            continue
        out.append(msgs)
        if max_per_conv and len(out) >= max_per_conv:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coser_dir", default="data/raw/coser/full")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max", type=int, default=0, help="cap total examples (0=all)")
    ap.add_argument("--max_per_conv", type=int, default=2)
    ap.add_argument("--max_books", type=int, default=0)
    args = ap.parse_args()

    books = sorted(glob.glob(os.path.join(args.coser_dir, "*.json")))
    if args.max_books:
        books = books[: args.max_books]

    sft_rows, verl_rows = [], []
    for fp in books:
        try:
            book = json.load(open(fp))
        except Exception:
            continue
        profiles = {n: d.get("profile", "")
                    for n, d in (book.get("character_datasets") or {}).items()}
        for plot in book.get("plots", []):
            for conv in plot.get("conversation", []) or []:
                if not isinstance(conv, dict):
                    continue
                for msgs in build_examples_for_conv(conv, profiles, args.max_per_conv):
                    sft_rows.append({"conversations": msgs})
                    history = msgs[:-1]
                    gold = msgs[-1]["value"]
                    prompt_messages = [{"role": role_map(m["from"]), "content": m["value"]}
                                       for m in history]
                    verl_rows.append({
                        "id": f"task2_coser_{len(verl_rows)}",
                        "data_source": "npc/roleplay",
                        "prompt": prompt_messages,
                        "ability": "roleplay",
                        "reward_model": {"ground_truth": gold, "style": "rule"},
                        "extra_info": {"task_name": "task2",
                                        "question": build_prompt_str(prompt_messages)},
                    })
                    if args.max and len(verl_rows) >= args.max:
                        break
                if args.max and len(verl_rows) >= args.max:
                    break
            if args.max and len(verl_rows) >= args.max:
                break
        if args.max and len(verl_rows) >= args.max:
            break

    os.makedirs(os.path.join(args.out, "sft"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "verl"), exist_ok=True)
    json.dump(sft_rows, open(os.path.join(args.out, "sft", "stage_1.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(verl_rows, open(os.path.join(args.out, "verl", "stage_1.json"), "w"),
              ensure_ascii=False, indent=2)
    pd.DataFrame(verl_rows).to_parquet(os.path.join(args.out, "verl", "stage_1.parquet"),
                                       engine="pyarrow", index=False)
    print(f"stage_1: {len(sft_rows)} SFT rows, {len(verl_rows)} verl rows from {len(books)} books")


if __name__ == "__main__":
    main()
