"""
Run check_reasoning on run_bench.py output files.

For each record that has a valid pred and llm_output (evidence_a, evidence_b, reason),
calls the LLM with check_reasoning_prompt.txt and records supported / critique.

Usage:
  python bench/evaluation/check_reasoning.py \
    --results results/new_prompt_full/positives/openai_gpt-5.4/full.json \
             results/new_prompt_full/negatives/openai_gpt-5.4/full.json \
    --output  results/reasoning_check_new_prompt_full.json \
    --base-url $NOVGAUGE_BASE_URL \
    --api-key  <your-key> \
    --model    openai/gpt-5.2 \
    --workers  8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).parent
REPO = BASE.parent.parent
sys.path.insert(0, str(REPO))

from bench.llm_client import LLMConfig, chat_complete_json_text

PROMPT_PATH = REPO / "bench" / "prompts" / "check_reasoning_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(dim: str, pred: int, evidence_a: str, evidence_b: str, reason: str) -> str:
    conclusion = "similar" if pred == 1 else "not similar"
    return (
        PROMPT_TEMPLATE
        .replace("<DIMENSION>", dim)
        .replace("<CONCLUSION>", conclusion)
        .replace("<EVIDENCE_A>", evidence_a or "(none)")
        .replace("<EVIDENCE_B>", evidence_b or "(none)")
        .replace("<REASON>", reason or "(none)")
    )


def call_llm(cfg: LLMConfig, prompt: str) -> tuple[bool | None, str]:
    """Returns (supported, critique). supported=None on parse failure."""
    raw = chat_complete_json_text(cfg=cfg, messages=[{"role": "user", "content": prompt}])
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return bool(obj["supported"]), str(obj.get("critique", ""))
    except Exception:
        return None, f"parse_error: {raw[:200]}"


def process_record(rec: dict, cfg: LLMConfig) -> dict:
    result = {
        "pair_id":    rec.get("pair_id"),
        "dim":        rec.get("dim"),
        "label":      rec.get("label"),
        "pred":       rec.get("pred"),
        "evidence_a": None,
        "evidence_b": None,
        "reason":     None,
        "conclusion": None,
        "supported":  None,
        "critique":   None,
        "error":      None,
    }

    pred = rec.get("pred")
    if pred != 1:
        result["error"] = "skipped_non_similar"
        return result

    samples = rec.get("samples") or []
    lo = samples[0].get("llm_output", {}) if samples else {}
    if not lo:
        result["error"] = "no_llm_output"
        return result

    ev_a_raw   = lo.get("evidence_a") or ""
    ev_b_raw   = lo.get("evidence_b") or ""
    evidence_a = ev_a_raw.strip() if isinstance(ev_a_raw, str) else ""
    evidence_b = ev_b_raw.strip() if isinstance(ev_b_raw, str) else ""
    reason_raw = lo.get("reason") or ""
    reason     = reason_raw.strip() if isinstance(reason_raw, str) else ""

    result["evidence_a"] = evidence_a
    result["evidence_b"] = evidence_b
    result["reason"]     = reason
    result["conclusion"] = "similar" if pred == 1 else "not similar"

    if not reason:
        result["error"] = "no_reason"
        return result

    prompt = build_prompt(rec["dim"], pred, evidence_a, evidence_b, reason)
    try:
        supported, critique = call_llm(cfg, prompt)
        result["supported"] = supported
        result["critique"]  = critique
    except Exception as e:
        result["error"] = str(e)

    return result


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", required=True,
                    help="run_bench.py output JSON file(s)")
    ap.add_argument("--output", required=True,
                    help="Output JSON file path")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip pair_id+dim already present in --output")
    ap.add_argument("--base-url",  required=True,  help="LLM API base URL")
    ap.add_argument("--api-key",   required=True,  help="LLM API key")
    ap.add_argument("--model",     default="openai/gpt-5.2", help="Judge model (default: openai/gpt-5.2)")
    args = ap.parse_args()

    cfg = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=0,
        max_tokens=512,
    )
    print(f"Model: {cfg.model}  Workers: {args.workers}")

    records = load_records([Path(p) for p in args.results])
    print(f"Loaded {len(records)} records")

    out_path = Path(args.output)
    existing: set[tuple] = set()
    done: list[dict] = []
    if args.skip_existing and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = json.load(f)
        existing = {(d["pair_id"], d["dim"]) for d in done}
        print(f"Skipping {len(existing)} already-done records")

    to_run = [r for r in records
              if (r.get("pair_id"), r.get("dim")) not in existing]
    print(f"To process: {len(to_run)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_record, r, cfg): r for r in to_run}
        for fut in as_completed(futures):
            res = fut.result()
            done.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(to_run):
                print(f"  {completed}/{len(to_run)}  "
                      f"supported={sum(1 for d in done if d.get('supported') is True)}  "
                      f"errors={sum(1 for d in done if d.get('error'))}")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(done, f, ensure_ascii=False, indent=2)

    # Final save + summary
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2)

    valid = [d for d in done if d.get("supported") is not None]
    errors = [d for d in done if d.get("error")]
    sup_true  = sum(1 for d in valid if d["supported"])
    sup_false = sum(1 for d in valid if not d["supported"])
    print(f"\nDone. Total={len(done)}  valid={len(valid)}  errors={len(errors)}")
    print(f"  supported=True:  {sup_true}  ({100*sup_true/len(valid):.1f}%)" if valid else "")
    print(f"  supported=False: {sup_false}  ({100*sup_false/len(valid):.1f}%)" if valid else "")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
