"""
Run check_reasoning on run_grouping_bench.py output files.

For each predicted group that is not hallucinated, calls the LLM judge with
check_reasoning_grouping_prompt.txt to verify whether the stated reason is
logically supported by the evidence strings.

Usage:
  python bench/evaluation/check_reasoning_grouping.py \\
    --results results/grouping_openai_gpt-5.4_full/openai_gpt-5.4_full.json \\
    --output  results/grouping_openai_gpt-5.4_full/eval_pipeline/reasoning_check_grouping.json \\
    --base-url $NOVGAUGE_BASE_URL \\
    --api-key  <your-key> \\
    --model    openai/gpt-5.2 \\
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

PROMPT_PATH     = REPO / "bench" / "prompts" / "check_reasoning_grouping_prompt.txt"
PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompt(dim: str, evidence_map: dict, reason: str) -> str:
    """Build the judge prompt for a single group.

    Args:
        dim:          "task" | "problem" | "method"
        evidence_map: {paper_index (str or int): evidence_text}
        reason:       The analyst's stated reason for the grouping.
    """
    # Build evidence blocks, one per paper
    blocks: list[str] = []
    def _sort_key(kv):
        try:
            return (0, int(kv[0]))
        except (ValueError, TypeError):
            return (1, str(kv[0]))

    for paper_idx, ev_text in sorted(evidence_map.items(), key=_sort_key):
        ev = (ev_text.strip() if isinstance(ev_text, str) else "") or "(none)"
        blocks.append(
            f"[Paper {paper_idx} — verbatim excerpt]\n{ev}\n[Paper {paper_idx} END]"
        )
    evidence_blocks = "\n\n".join(blocks) if blocks else "(no evidence provided)"

    return (
        PROMPT_TEMPLATE
        .replace("<DIM>", dim)
        .replace("<EVIDENCE_BLOCKS>", evidence_blocks)
        .replace("<REASON>", (reason if isinstance(reason, str) else "") or "(none)")
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(cfg: LLMConfig, prompt: str) -> tuple[bool | None, str]:
    """Returns (supported, critique). supported=None on parse failure."""
    raw = chat_complete_json_text(cfg=cfg, messages=[{"role": "user", "content": prompt}])
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return bool(obj["supported"]), str(obj.get("critique", ""))
    except Exception:
        return None, f"parse_error: {raw[:200]}"


# ── Per-group worker ──────────────────────────────────────────────────────────

def process_group(record: dict, group_idx: int, cfg: LLMConfig) -> dict:
    """Process a single predicted group from a run_grouping_bench result record.

    Args:
        record:    A result dict from run_grouping_bench.py output["results"].
        group_idx: Index into record["predicted_groups_rich"].
        cfg:       LLMConfig for the judge model.

    Returns a dict with keys:
        item_id, dim, group_idx, paper_indices, supported, critique, error
    """
    item_id = record.get("item_id", "")
    dim     = record.get("dim", "")

    result: dict = {
        "item_id":       item_id,
        "dim":           dim,
        "group_idx":     group_idx,
        "paper_indices": [],
        "supported":     None,
        "critique":      None,
        "error":         None,
    }

    rich_groups = record.get("predicted_groups_rich") or []
    if group_idx >= len(rich_groups):
        result["error"] = f"group_idx {group_idx} out of range (n_groups={len(rich_groups)})"
        return result

    grp = rich_groups[group_idx]
    paper_indices = grp.get("paper_indices", [])
    evidence_raw  = grp.get("evidence", {})
    evidence_map  = evidence_raw if isinstance(evidence_raw, dict) else {}
    reason_raw    = grp.get("reason") or ""
    reason        = reason_raw.strip() if isinstance(reason_raw, str) else ""

    result["paper_indices"] = paper_indices
    result["reason"]        = reason
    result["evidence_map"]  = dict(evidence_map)

    if not reason:
        result["error"] = "no_reason"
        return result

    if not evidence_map:
        result["error"] = "no_evidence"
        return result

    prompt = build_prompt(dim, evidence_map, reason)
    try:
        supported, critique = call_llm(cfg, prompt)
        result["supported"] = supported
        result["critique"]  = critique
    except Exception as e:
        result["error"] = str(e)

    return result


# ── Load records ──────────────────────────────────────────────────────────────

def load_records(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("results", [])
    return data


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results", required=True,
                    help="run_grouping_bench.py output JSON file")
    ap.add_argument("--output",  required=True,
                    help="Output JSON file path")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip (item_id, dim, group_idx) already present in --output")
    ap.add_argument("--base-url",  required=True,  help="LLM API base URL")
    ap.add_argument("--api-key",   required=True,  help="LLM API key")
    ap.add_argument("--model",     default="openai/gpt-5.2",
                    help="Judge model (default: openai/gpt-5.2)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens",  type=int,   default=512)
    args = ap.parse_args()

    cfg = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"Model: {cfg.model}  Workers: {args.workers}")

    records = load_records(Path(args.results))
    print(f"Loaded {len(records)} records from {args.results}")

    # Build (item_id, dim, group_idx) work items
    work_items: list[tuple[dict, int]] = []
    for rec in records:
        rich_groups = rec.get("predicted_groups_rich") or []
        for grp_idx in range(len(rich_groups)):
            work_items.append((rec, grp_idx))
    print(f"Total groups to check: {len(work_items)}")

    out_path = Path(args.output)
    existing: set[tuple] = set()
    done: list[dict] = []
    if args.skip_existing and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = json.load(f)
        existing = {(d["item_id"], d["dim"], d["group_idx"]) for d in done}
        print(f"Skipping {len(existing)} already-done groups")

    to_run = [
        (rec, grp_idx) for rec, grp_idx in work_items
        if (rec.get("item_id"), rec.get("dim"), grp_idx) not in existing
    ]
    print(f"To process: {len(to_run)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_group, rec, grp_idx, cfg): (rec, grp_idx)
                   for rec, grp_idx in to_run}
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
    if valid:
        print(f"  supported=True:  {sup_true}  ({100*sup_true/len(valid):.1f}%)")
        print(f"  supported=False: {sup_false}  ({100*sup_false/len(valid):.1f}%)")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
