#!/usr/bin/env python3
"""Add the hint and guide fields to a BSD trace parquet (the BSD-hint and BSD-guide variants).

For every `--form NAME=path.jsonl` (JSONL from generate_pi_forms.py, one row per uid) a field
`<trace-field-prefix>_<NAME>` is added to extra_info next to the untouched source trace. With --append-answer
the text becomes "<body>\n\nFinal answer: <ground_truth>", so every form ends with the verified final answer,
as the full trace does. Rows missing from a JSONL are an error. Writes the parquet and a manifest with the
leak statistics and hashes. Run with the training interpreter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ANSWER_LINE = "\n\nFinal answer: {gt}"


def load_form(path: str) -> dict[str, dict]:
    recs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[str(r["uid"])] = r
    return recs


def build(df: pd.DataFrame, forms: dict[str, dict[str, dict]], prefix: str, append_answer: bool) -> tuple[pd.DataFrame, dict]:
    infos, stats = [], {name: {"rows": 0, "answer_leak": 0, "candidates": 0, "empty": 0, "chars_total": 0} for name in forms}
    for e, rm in zip(df["extra_info"], df["reward_model"]):
        e = dict(e)
        uid = str(e["uid"])
        for name, recs in forms.items():
            if uid not in recs:
                raise KeyError(f"form {name}: no text for uid {uid}")
            body = str(recs[uid]["text"]).strip()
            if not body:
                stats[name]["empty"] += 1
            text = body + ANSWER_LINE.format(gt=str(rm["ground_truth"])) if append_answer else body
            e[f"{prefix}_{name}"] = text
            s = stats[name]
            s["rows"] += 1; s["answer_leak"] += bool(recs[uid].get("answer_leak")); s["candidates"] += int(recs[uid].get("n_candidates", 1))
            s["chars_total"] += len(text)
        infos.append(e)
    out = df.copy()
    out["extra_info"] = infos
    return out, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--form", action="append", required=True, metavar="NAME=JSONL")
    p.add_argument("--trace-field", default="qwen32b_pi_trace", help="source trace field; new fields are <field minus _trace>_<name>")
    p.add_argument("--append-answer", action="store_true", help="append 'Final answer: <gt>' to every form")
    args = p.parse_args()
    prefix = args.trace_field[: -len("_trace")] if args.trace_field.endswith("_trace") else args.trace_field
    forms = {}
    for spec in args.form:
        name, path = spec.split("=", 1)
        forms[name] = load_form(path)
    df = pd.read_parquet(args.input)
    out, stats = build(df, forms, prefix, args.append_answer)
    if any(s["empty"] for s in stats.values()):
        raise ValueError(f"empty PI text: { {n: s['empty'] for n, s in stats.items()} }")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    models = sorted({str(r.get("model")) for recs in forms.values() for r in recs.values()})
    manifest = {
        "source": args.input, "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "output": args.output, "output_sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
        "rows": len(out), "fields_added": [f"{prefix}_{n}" for n in forms], "append_answer": args.append_answer,
        "answer_line": ANSWER_LINE.strip() if args.append_answer else None, "generator_models": models,
        "forms": {n: dict(s, jsonl=[spec.split('=', 1)[1] for spec in args.form if spec.startswith(n + '=')][0],
                          chars_mean=round(s["chars_total"] / max(s["rows"], 1)), leak_rate=round(s["answer_leak"] / max(s["rows"], 1), 4))
                  for n, s in stats.items()},
        "builder": "tools/bsd/build_pi_form_views.py",
    }
    Path(args.output).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
