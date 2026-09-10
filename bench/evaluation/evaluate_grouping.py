"""Evaluate grouping predictions through similarity, evidence, reasoning, and Verified F1 stages."""
from __future__ import annotations

import argparse
import os
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).parent
REPO = BASE.parent.parent
sys.path.insert(0, str(REPO))

from bench.evaluation.check_evidence_grouping import (
    check_file as check_evidence_grouping_file,
    _build_lookup,
)
from bench.evaluation.check_reasoning_grouping import process_group
from bench.llm_client import LLMConfig

DIMS = ["task", "problem", "method"]


# helpers
def load_json(path: str | Path) -> object:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:.1f}%" if denom else "n/a"


def load_records(paths: list[str | Path]) -> list[dict]:
    """Load result records from one or more run_grouping_bench output files."""
    records: list[dict] = []
    for p in paths:
        data = load_json(p)
        if isinstance(data, dict):
            records.extend(data.get("results", []))
        else:
            records.extend(data)
    return records


# Pairwise eval helpers (mirrors run_grouping_bench.py)
def groups_to_pairs(groups: list[list[int]]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for grp in groups:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                pairs.add((min(grp[i], grp[j]), max(grp[i], grp[j])))
    return pairs


def eval_groups(pred_groups: list[list[int]], gt_groups: list[list[int]]) -> dict:
    pred_pairs = groups_to_pairs(pred_groups)
    gt_pairs   = groups_to_pairs(gt_groups)
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs  - pred_pairs)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall":    round(r, 4),
        "f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "pred_pairs": sorted(pred_pairs),
        "gt_pairs":   sorted(gt_pairs),
    }


# Stage 1: grouping accuracy
def stage1_accuracy(records: list[dict]) -> dict:
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 1 — Grouping Accuracy  (pairwise F1 per dim)")
    print("=" * 65)

    # Detect multi-sample: any record has sample_idx > 0
    sample_indices = sorted(set(r.get("sample_idx", 0) for r in records))
    is_multi_sample = len(sample_indices) > 1

    dim_metrics: dict[str, dict] = {}

    for dim in DIMS:
        sub = [r for r in records if r.get("dim") == dim and r.get("eval_valid")]
        if not sub:
            print(f"\n  [{dim}]  no eval items")
            dim_metrics[dim] = {}
            continue

        if is_multi_sample:
            # Per-sample macro/micro metrics, then mean±std across samples
            sample_macro_f1, sample_macro_p, sample_macro_r = [], [], []
            sample_micro_f1 = []
            for s_idx in sample_indices:
                evaled_s = [r for r in sub
                            if r.get("sample_idx", 0) == s_idx and r.get("metrics")]
                if not evaled_s:
                    continue
                sp = sum(r["metrics"]["precision"] for r in evaled_s) / len(evaled_s)
                sr = sum(r["metrics"]["recall"]    for r in evaled_s) / len(evaled_s)
                sf = sum(r["metrics"]["f1"]        for r in evaled_s) / len(evaled_s)
                sample_macro_p.append(sp)
                sample_macro_r.append(sr)
                sample_macro_f1.append(sf)
                tp = sum(r["metrics"]["tp"] for r in evaled_s)
                fp = sum(r["metrics"]["fp"] for r in evaled_s)
                fn = sum(r["metrics"]["fn"] for r in evaled_s)
                mi_p = tp / (tp + fp) if (tp + fp) else 0.0
                mi_r = tp / (tp + fn) if (tp + fn) else 0.0
                mi_f = 2 * mi_p * mi_r / (mi_p + mi_r) if (mi_p + mi_r) else 0.0
                sample_micro_f1.append(mi_f)

            n_items_per_sample = len([r for r in sub if r.get("sample_idx", 0) == sample_indices[0]])
            mean_macro_f1 = _stats.mean(sample_macro_f1) if sample_macro_f1 else 0.0
            std_macro_f1  = _stats.stdev(sample_macro_f1) if len(sample_macro_f1) > 1 else 0.0
            mean_macro_p  = _stats.mean(sample_macro_p)  if sample_macro_p  else 0.0
            mean_macro_r  = _stats.mean(sample_macro_r)  if sample_macro_r  else 0.0
            mean_micro_f1 = _stats.mean(sample_micro_f1) if sample_micro_f1 else 0.0
            std_micro_f1  = _stats.stdev(sample_micro_f1) if len(sample_micro_f1) > 1 else 0.0

            print(f"\n  [{dim}]  n_items={n_items_per_sample}  n_samples={len(sample_indices)}")
            print(f"    {'':12}  {'Precision':>10}  {'Recall':>8}  {'F1 mean':>10}  {'F1 std':>8}")
            print(f"    {'Macro-avg':<12}  {mean_macro_p:>10.3f}  {mean_macro_r:>8.3f}  "
                  f"{mean_macro_f1:>10.3f}  ±{std_macro_f1:.3f}")
            print(f"    {'Micro-avg':<12}  {'':>10}  {'':>8}  "
                  f"{mean_micro_f1:>10.3f}  ±{std_micro_f1:.3f}")
            print(f"    per-sample macro-F1: {[round(f, 3) for f in sample_macro_f1]}")
            print(f"    per-sample micro-F1: {[round(f, 3) for f in sample_micro_f1]}")
            dim_metrics[dim] = {
                "n_items": n_items_per_sample,
                "n_samples": len(sample_indices),
                "macro_precision_mean": round(mean_macro_p, 4),
                "macro_recall_mean": round(mean_macro_r, 4),
                "macro_f1_mean": round(mean_macro_f1, 4),
                "macro_f1_std": round(std_macro_f1, 4),
                "micro_f1_mean": round(mean_micro_f1, 4),
                "micro_f1_std": round(std_micro_f1, 4),
            }
        else:
            evaled = [r for r in sub if r.get("metrics") is not None]
            if not evaled:
                print(f"\n  [{dim}]  no eval items")
                dim_metrics[dim] = {}
                continue

            parse_errs = sum(1 for r in sub if r.get("parse_error"))
            llm_errs   = sum(1 for r in sub if r.get("llm_error"))

            macro_p  = sum(r["metrics"]["precision"] for r in evaled) / len(evaled)
            macro_r  = sum(r["metrics"]["recall"]    for r in evaled) / len(evaled)
            macro_f1 = sum(r["metrics"]["f1"]        for r in evaled) / len(evaled)

            total_tp = sum(r["metrics"]["tp"] for r in evaled)
            total_fp = sum(r["metrics"]["fp"] for r in evaled)
            total_fn = sum(r["metrics"]["fn"] for r in evaled)
            micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
            micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
            micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                        if (micro_p + micro_r) else 0.0)

            print(f"\n  [{dim}]  n_items={len(sub)}  n_eval={len(evaled)}"
                  f"  parse_err={parse_errs}  llm_err={llm_errs}")
            print(f"    {'':12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
            print(f"    {'Macro-avg':<12}  {macro_p:>10.3f}  {macro_r:>8.3f}  {macro_f1:>8.3f}")
            print(f"    {'Micro-avg':<12}  {micro_p:>10.3f}  {micro_r:>8.3f}  {micro_f1:>8.3f}"
                  f"  (TP={total_tp} FP={total_fp} FN={total_fn})")
            dim_metrics[dim] = {
                "n_items": len(sub),
                "n_eval": len(evaled),
                "parse_err": parse_errs,
                "llm_err": llm_errs,
                "macro_precision": round(macro_p, 4),
                "macro_recall": round(macro_r, 4),
                "macro_f1": round(macro_f1, 4),
                "micro_precision": round(micro_p, 4),
                "micro_recall": round(micro_r, 4),
                "micro_f1": round(micro_f1, 4),
                "tp": total_tp, "fp": total_fp, "fn": total_fn,
            }

    return {"dims": dim_metrics}


# Stage 2: hallucination rate
def _compute_halluc_for_sample(
    sample_recs: list[dict],
    item_lookup: dict[str, dict],
    mode: str,
    tmp_path: Path,
) -> tuple[dict, set[tuple[str, str, int]]]:
    """Run check_evidence_grouping on one sample's records; return (stats, hallucinated_set)."""
    save_json(sample_recs, tmp_path)
    try:
        stats = check_evidence_grouping_file(tmp_path, item_lookup, mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return stats, stats.get("hallucinated_set", set())


def stage2_hallucination(
    records: list[dict],
    item_lookup: dict[str, dict],
    mode: str,
    output_dir: Path,
) -> tuple[list[set[tuple[str, str, int]]], dict]:
    """Check evidence quality per sample.

    Returns (per_sample_hallucinated, metrics_dict).
    """
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 2 — Hallucination Rate  (predicted groups)")
    print("=" * 65)

    sample_indices = sorted(set(r.get("sample_idx", 0) for r in records))
    is_multi = len(sample_indices) > 1
    if is_multi:
        print(f"  [multi-sample] n_samples={len(sample_indices)}, computing per-sample rates")

    per_sample_hallucinated: list[set[tuple[str, str, int]]] = []
    all_details: list[dict] = []
    sample_halluc_rates: list[float] = []
    stage2_metrics: dict = {}

    for s_idx in sample_indices:
        sample_recs = [r for r in records if r.get("sample_idx", 0) == s_idx]
        tmp_path = output_dir / f"_tmp_grouping_s{s_idx}.json"
        stats, hallucinated_s = _compute_halluc_for_sample(
            sample_recs, item_lookup, mode, tmp_path
        )
        per_sample_hallucinated.append(hallucinated_s)
        all_details.extend(stats.get("details", []))

        n     = stats["total_groups"]
        halluc = stats["hallucination"]
        rate   = halluc / n if n else 0.0
        sample_halluc_rates.append(rate)

        if is_multi:
            print(f"\n  ── sample {s_idx}  (groups={n})")
            if n:
                print(f"    Verbatim      : {stats['verbatim']:4d} / {n}  ({pct(stats['verbatim'], n)})")
                print(f"    Artifact-only : {stats['artifact']:4d} / {n}  ({pct(stats['artifact'], n)})")
                print(f"    Hallucination : {halluc:4d} / {n}  ({pct(halluc, n)})")
        else:
            # Single-sample: full detailed output
            print(f"\n  Total predicted groups: {n}")
            details = stats.get("details", [])
            dim_metrics: dict[str, dict] = {}
            if n:
                print(f"  Verbatim      : {stats['verbatim']:4d} / {n}  ({pct(stats['verbatim'], n)})")
                print(f"  Artifact-only : {stats['artifact']:4d} / {n}  ({pct(stats['artifact'], n)})")
                print(f"  Hallucination : {halluc:4d}   / {n}  ({pct(halluc, n)})")

                print(f"\n  Per-dimension breakdown:")
                print(f"  {'dim':<10} {'groups':>7} {'halluc%':>10}")
                print(f"  {'-'*32}")
                for dim in DIMS:
                    dim_groups = {(d["item_id"], d["dim"], d["group_idx"])
                                  for d in details if d["dim"] == dim}
                    dim_n  = len(dim_groups)
                    dim_ha = sum(1 for k in hallucinated_s if k[1] == dim)
                    print(f"  {dim:<10} {dim_n:>7}   {pct(dim_ha, dim_n):>9}")
                    dim_metrics[dim] = {
                        "n_groups": dim_n,
                        "hallucination": dim_ha,
                        "halluc_rate": round(dim_ha / dim_n * 100, 2) if dim_n else 0,
                    }

                # Correct vs incorrect
                gt_pairs_map: dict[tuple[str, str], set] = {}
                for rec in sample_recs:
                    key = (rec.get("item_id", ""), rec.get("dim", ""))
                    if key not in gt_pairs_map:
                        gt_pairs_map[key] = groups_to_pairs(rec.get("ground_truth_groups") or [])
                c_tot = c_ha = i_tot = i_ha = 0
                for rec in sample_recs:
                    iid = rec.get("item_id", "")
                    dim = rec.get("dim", "")
                    gt_pairs = gt_pairs_map.get((iid, dim), set())
                    for gi, grp in enumerate(rec.get("predicted_groups_rich") or []):
                        pp = groups_to_pairs([grp.get("paper_indices", [])])
                        is_correct = pp.issubset(gt_pairs)
                        is_ha = (iid, dim, gi) in hallucinated_s
                        if is_correct:
                            c_tot += 1; c_ha += is_ha
                        else:
                            i_tot += 1; i_ha += is_ha
                print(f"\n  Correct vs incorrect predicted groups:")
                print(f"  {'group_type':<25} {'total':>7} {'halluc':>7} {'halluc%':>10}")
                print(f"  {'-'*52}")
                print(f"  {'correct (all pairs in GT)':<25} {c_tot:>7} {c_ha:>7} {pct(c_ha, c_tot):>10}")
                print(f"  {'incorrect (has FP pairs)':<25} {i_tot:>7} {i_ha:>7} {pct(i_ha, i_tot):>10}")

            stage2_metrics = {
                "total_groups": n,
                "verbatim": stats.get("verbatim", 0),
                "artifact": stats.get("artifact", 0),
                "hallucination": halluc,
                "halluc_rate": round(rate * 100, 2),
                "dims": dim_metrics,
                "correct_groups": {"total": c_tot, "hallucination": c_ha, "halluc_rate": round(c_ha / c_tot * 100, 2) if c_tot else 0},
                "incorrect_groups": {"total": i_tot, "hallucination": i_ha, "halluc_rate": round(i_ha / i_tot * 100, 2) if i_tot else 0},
            } if n else {"total_groups": 0}

    if is_multi and sample_halluc_rates:
        m = _stats.mean(sample_halluc_rates) * 100
        s = (_stats.stdev(sample_halluc_rates) * 100) if len(sample_halluc_rates) > 1 else 0.0
        print(f"\n  Mean±std hallucination rate: {m:.1f}% ± {s:.1f}%")
        stage2_metrics = {
            "halluc_rate_mean": round(m, 2),
            "halluc_rate_std": round(s, 2),
            "per_sample": sample_halluc_rates,
        }

    if all_details:
        halluc_details = [d for d in all_details if d.get("group_cls") == "hallucination"]
        save_json(halluc_details, output_dir / "hallucinations_grouping.json")
        print(f"\n  Hallucination details → {output_dir / 'hallucinations_grouping.json'}")

    return per_sample_hallucinated, stage2_metrics


# Stage 3: mismatch rate
def stage3_mismatch(
    records: list[dict],
    per_sample_hallucinated: list[set[tuple[str, str, int]]],
    cfg: LLMConfig,
    workers: int,
    output_dir: Path,
    skip_existing: bool = True,
    item_lookup: dict | None = None,
) -> dict:
    """Check reasoning quality for non-hallucinated predicted groups, per sample."""
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 3 — Mismatch Rate  (non-hallucinated predicted groups)")
    print("=" * 65)

    sample_indices = sorted(set(r.get("sample_idx", 0) for r in records))
    is_multi = len(sample_indices) > 1
    if is_multi:
        print(f"  [multi-sample] n_samples={len(sample_indices)}, computing per-sample rates")

    # Build all (rec, grp_idx, s_idx) candidates across all samples
    to_run_all: list[tuple[dict, int, int]] = []
    total_groups = 0
    # gt_pairs per (item_id, dim, sample_idx) for is_correct_group
    gt_pairs_lookup: dict[tuple[str, str, int], set] = {}
    for s_idx, hallucinated_s in zip(sample_indices, per_sample_hallucinated):
        sample_recs = [r for r in records if r.get("sample_idx", 0) == s_idx]
        for rec in sample_recs:
            item_id = rec.get("item_id", "")
            dim     = rec.get("dim", "")
            key     = (item_id, dim, s_idx)
            if key not in gt_pairs_lookup:
                gt_pairs_lookup[key] = groups_to_pairs(rec.get("ground_truth_groups") or [])
            rich_groups = rec.get("predicted_groups_rich") or []
            for grp_idx in range(len(rich_groups)):
                total_groups += 1
                if (item_id, dim, grp_idx) not in hallucinated_s:
                    to_run_all.append((rec, grp_idx, s_idx))

    print(f"\n  Total predicted groups (all samples)     : {total_groups}")
    print(f"  Non-hallucinated candidates (all samples): {len(to_run_all)}")

    out_path = output_dir / "reasoning_check_grouping.json"
    done: list[dict] = []
    existing: set[tuple] = set()
    if skip_existing and out_path.exists():
        done = load_json(out_path)  # type: ignore[assignment]
        existing = {
            (d["item_id"], d["dim"], d["group_idx"], d.get("sample_idx", 0))
            for d in done if d.get("supported") is not None
        }
        done = [d for d in done if d.get("supported") is not None]
        print(f"  Skipping {len(existing)} already done")

    to_run = [
        (rec, grp_idx, s_idx) for rec, grp_idx, s_idx in to_run_all
        if (rec.get("item_id"), rec.get("dim"), grp_idx, s_idx) not in existing
    ]
    print(f"  To run                                   : {len(to_run)}")

    def _process(rec: dict, grp_idx: int, s_idx: int) -> dict:
        result = process_group(rec, grp_idx, cfg)
        result["sample_idx"] = s_idx
        # Annotate is_correct_group
        iid  = rec.get("item_id", "")
        dim  = rec.get("dim", "")
        rich = (rec.get("predicted_groups_rich") or [])
        if grp_idx < len(rich):
            pp = groups_to_pairs([rich[grp_idx].get("paper_indices", [])])
            gt = gt_pairs_lookup.get((iid, dim, s_idx), set())
            result["is_correct_group"] = pp.issubset(gt) if gt else None
        else:
            result["is_correct_group"] = None
        # Annotate paper titles from item_lookup
        if item_lookup:
            item = item_lookup.get(iid, {})
            paper_by_idx = {p["paper_index"]: p for p in item.get("papers", [])}
            result["paper_titles"] = {
                str(pidx): paper_by_idx.get(pidx, {}).get("title", "")
                for pidx in result.get("paper_indices", [])
            }
        return result

    if to_run:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process, rec, grp_idx, s_idx): (rec, grp_idx, s_idx)
                for rec, grp_idx, s_idx in to_run
            }
            for fut in as_completed(futures):
                done.append(fut.result())
                completed += 1
                if completed % 50 == 0 or completed == len(to_run):
                    print(f"    {completed}/{len(to_run)}  "
                          f"supported={sum(1 for d in done if d.get('supported') is True)}")
                    save_json(done, out_path)

    save_json(done, out_path)

    # Build GT pairs lookup for correct/incorrect classification
    gt_pairs_lookup: dict[tuple[str, str, int], set] = {}
    for rec in records:
        iid = rec.get("item_id", "")
        dim = rec.get("dim", "")
        s   = rec.get("sample_idx", 0)
        gt_pairs_lookup[(iid, dim, s)] = groups_to_pairs(rec.get("ground_truth_groups") or [])

    def _is_correct_group(d: dict) -> bool:
        """True if all pairs in the predicted group are GT pairs."""
        key = (d.get("item_id", ""), d.get("dim", ""), d.get("sample_idx", 0))
        gt_pairs = gt_pairs_lookup.get(key, set())
        pp = groups_to_pairs([d.get("paper_indices", [])])
        return pp.issubset(gt_pairs)

    # Per-sample summary
    print(f"\n  Results  (supported=True → reasoning verified, False → mismatch)")
    sample_mismatch_rates: list[float] = []
    sample_correct_mismatch_rates: list[float] = []
    sample_incorrect_mismatch_rates: list[float] = []

    for s_idx in sample_indices:
        valid = [d for d in done
                 if d.get("sample_idx", 0) == s_idx and d.get("supported") is not None]
        if not valid:
            continue

        sup_f_all = sum(1 for d in valid if not d["supported"])
        sup_t_all = sum(1 for d in valid if d["supported"])
        sample_mismatch_rates.append(sup_f_all / len(valid))

        correct_v   = [d for d in valid if _is_correct_group(d)]
        incorrect_v = [d for d in valid if not _is_correct_group(d)]
        c_mm = sum(1 for d in correct_v   if not d["supported"])
        i_mm = sum(1 for d in incorrect_v if not d["supported"])
        if correct_v:
            sample_correct_mismatch_rates.append(c_mm / len(correct_v))
        if incorrect_v:
            sample_incorrect_mismatch_rates.append(i_mm / len(incorrect_v))

        indent = "    " if is_multi else "  "
        if is_multi:
            print(f"\n  ── sample {s_idx}  (groups checked={len(valid)})")
        print(f"{indent}{'dim':<10} {'checked':>8} {'True':>8} {'False':>8} {'mismatch%':>12}")
        print(f"{indent}{'-'*50}")
        for dim in DIMS:
            sub = [d for d in valid if d.get("dim") == dim]
            if not sub:
                continue
            sup_t = sum(1 for d in sub if d["supported"])
            sup_f = sum(1 for d in sub if not d["supported"])
            print(f"{indent}{dim:<10} {len(sub):>8} {sup_t:>8} {sup_f:>8} {pct(sup_f, len(sub)):>12}")
        print(f"{indent}{'ALL':<10} {len(valid):>8} {sup_t_all:>8} {sup_f_all:>8} "
              f"{pct(sup_f_all, len(valid)):>12}")
        print(f"{indent}Correct groups  : checked={len(correct_v)}  mismatch={c_mm}  "
              f"({pct(c_mm, len(correct_v))})")
        print(f"{indent}Incorrect groups: checked={len(incorrect_v)}  mismatch={i_mm}  "
              f"({pct(i_mm, len(incorrect_v))})")

    if is_multi and sample_mismatch_rates:
        m = _stats.mean(sample_mismatch_rates) * 100
        s = (_stats.stdev(sample_mismatch_rates) * 100) if len(sample_mismatch_rates) > 1 else 0.0
        each = [f"{r * 100:.1f}%" for r in sample_mismatch_rates]
        print(f"\n  Mean±std mismatch rate (overall)  : {m:.1f}% ± {s:.1f}%  per-sample={each}")
        if sample_correct_mismatch_rates:
            cm = _stats.mean(sample_correct_mismatch_rates) * 100
            cs = (_stats.stdev(sample_correct_mismatch_rates) * 100) if len(sample_correct_mismatch_rates) > 1 else 0.0
            print(f"  Mean±std mismatch rate (correct)  : {cm:.1f}% ± {cs:.1f}%")
        if sample_incorrect_mismatch_rates:
            im = _stats.mean(sample_incorrect_mismatch_rates) * 100
            is_ = (_stats.stdev(sample_incorrect_mismatch_rates) * 100) if len(sample_incorrect_mismatch_rates) > 1 else 0.0
            print(f"  Mean±std mismatch rate (incorrect): {im:.1f}% ± {is_:.1f}%")

    print(f"\n  Reasoning check results → {out_path}")

    # Build return metrics (single-sample case)
    stage3_metrics: dict = {}
    if not is_multi:
        valid_all = [d for d in done if d.get("supported") is not None]
        if valid_all:
            sup_t = sum(1 for d in valid_all if d["supported"])
            sup_f = sum(1 for d in valid_all if not d["supported"])
            correct_all   = [d for d in valid_all if _is_correct_group(d)]
            incorrect_all = [d for d in valid_all if not _is_correct_group(d)]
            c_mm_all = sum(1 for d in correct_all   if not d["supported"])
            i_mm_all = sum(1 for d in incorrect_all if not d["supported"])
            dim_m: dict[str, dict] = {}
            for dim in DIMS:
                sub = [d for d in valid_all if d.get("dim") == dim]
                if sub:
                    st = sum(1 for d in sub if d["supported"])
                    sf = sum(1 for d in sub if not d["supported"])
                    dim_m[dim] = {
                        "checked": len(sub),
                        "supported": st,
                        "mismatch": sf,
                        "mismatch_rate": round(sf / len(sub) * 100, 2),
                    }
            stage3_metrics = {
                "total_groups": total_groups,
                "non_hallucinated": len(to_run_all),
                "checked": len(valid_all),
                "supported": sup_t,
                "mismatch": sup_f,
                "mismatch_rate": round(sup_f / len(valid_all) * 100, 2) if valid_all else 0,
                "correct_groups": {
                    "checked": len(correct_all),
                    "mismatch": c_mm_all,
                    "mismatch_rate": round(c_mm_all / len(correct_all) * 100, 2) if correct_all else 0,
                },
                "incorrect_groups": {
                    "checked": len(incorrect_all),
                    "mismatch": i_mm_all,
                    "mismatch_rate": round(i_mm_all / len(incorrect_all) * 100, 2) if incorrect_all else 0,
                },
                "dims": dim_m,
            }
    elif sample_mismatch_rates:
        stage3_metrics = {
            "mismatch_rate_mean": round(_stats.mean(sample_mismatch_rates) * 100, 2),
            "mismatch_rate_std": round((_stats.stdev(sample_mismatch_rates) * 100) if len(sample_mismatch_rates) > 1 else 0.0, 2),
        }
        if sample_correct_mismatch_rates:
            stage3_metrics["correct_mismatch_mean"] = round(_stats.mean(sample_correct_mismatch_rates) * 100, 2)
            stage3_metrics["correct_mismatch_std"]  = round((_stats.stdev(sample_correct_mismatch_rates) * 100) if len(sample_correct_mismatch_rates) > 1 else 0.0, 2)
        if sample_incorrect_mismatch_rates:
            stage3_metrics["incorrect_mismatch_mean"] = round(_stats.mean(sample_incorrect_mismatch_rates) * 100, 2)
            stage3_metrics["incorrect_mismatch_std"]  = round((_stats.stdev(sample_incorrect_mismatch_rates) * 100) if len(sample_incorrect_mismatch_rates) > 1 else 0.0, 2)
    return stage3_metrics


# Stage 4: verified F1
def stage4_verified_f1(
    records: list[dict],
    per_sample_hallucinated: list[set[tuple[str, str, int]]],
    output_dir: Path,
) -> dict:
    """Compute Verified F1 from pairs in groups that pass evidence and reasoning checks.
    Unverified predicted pairs remain in the scoring denominators."""
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 4 — Verified F1  (grouping)")
    print("=" * 65)

    # Load Stage 3 results: (item_id, dim, group_idx, sample_idx) → supported
    reasoning_path = output_dir / "reasoning_check_grouping.json"
    reasoning_lookup: dict[tuple[str, str, int, int], bool | None] = {}
    if reasoning_path.exists():
        for d in load_json(reasoning_path):
            key = (d["item_id"], d["dim"], d["group_idx"], d.get("sample_idx", 0))
            reasoning_lookup[key] = d.get("supported")

    sample_indices = sorted(set(r.get("sample_idx", 0) for r in records))
    n_samp   = len(sample_indices)
    is_multi = n_samp > 1

    sample_results: list[dict[str, dict]] = []

    for s_idx, hallucinated_s in zip(sample_indices, per_sample_hallucinated):
        sample_recs = [r for r in records if r.get("sample_idx", 0) == s_idx]
        dim_m: dict[str, dict] = {}

        for dim in DIMS:
            sub = [r for r in sample_recs if r.get("dim") == dim and r.get("eval_valid")]
            if not sub:
                continue

            tp_verified = tp_halluc = tp_mismatch = tp_no_check = 0
            fp = fn = 0

            for rec in sub:
                item_id   = rec.get("item_id", "")
                gt_groups  = rec.get("ground_truth_groups") or []
                rich_groups = rec.get("predicted_groups_rich") or []
                gt_pairs   = groups_to_pairs(gt_groups)

                # Classify each predicted group
                verified_pairs: set[tuple[int, int]] = set()
                halluc_pairs:   set[tuple[int, int]] = set()
                mismatch_pairs: set[tuple[int, int]] = set()
                no_check_pairs: set[tuple[int, int]] = set()

                for grp_idx, grp in enumerate(rich_groups):
                    paper_indices = grp.get("paper_indices", [])
                    grp_pairs = groups_to_pairs([paper_indices])
                    if (item_id, dim, grp_idx) in hallucinated_s:
                        halluc_pairs |= grp_pairs
                    else:
                        sup = reasoning_lookup.get((item_id, dim, grp_idx, s_idx))
                        if sup is True:
                            verified_pairs |= grp_pairs
                        elif sup is False:
                            mismatch_pairs |= grp_pairs
                        else:
                            no_check_pairs |= grp_pairs

                all_pred_pairs = verified_pairs | halluc_pairs | mismatch_pairs | no_check_pairs

                tp_verified += len(verified_pairs & gt_pairs)
                tp_halluc   += len(halluc_pairs   & gt_pairs)
                tp_mismatch += len(mismatch_pairs & gt_pairs)
                tp_no_check += len(no_check_pairs & gt_pairs)
                fp          += len(all_pred_pairs - gt_pairs)
                fn          += len(gt_pairs - all_pred_pairs)

            tp_total = tp_verified + tp_halluc + tp_mismatch + tp_no_check
            v_p  = tp_verified / (tp_total + fp) if (tp_total + fp) else 0.0
            v_r  = tp_verified / (tp_total + fn) if (tp_total + fn) else 0.0
            v_f1 = 2 * v_p * v_r / (v_p + v_r) if (v_p + v_r) else 0.0
            dim_m[dim] = {
                "tp_verified":  tp_verified,
                "tp_halluc":    tp_halluc,
                "tp_mismatch":  tp_mismatch,
                "tp_no_check":  tp_no_check,
                "fp": fp, "fn": fn,
                "verified_precision": round(v_p, 4),
                "verified_recall":    round(v_r, 4),
                "verified_f1":        round(v_f1, 4),
            }
        sample_results.append(dim_m)

    all_dim_metrics: dict = {}
    for dim in DIMS:
        per_f1 = [m[dim]["verified_f1"]        for m in sample_results if dim in m]
        per_p  = [m[dim]["verified_precision"]  for m in sample_results if dim in m]
        per_r  = [m[dim]["verified_recall"]     for m in sample_results if dim in m]
        if not per_f1:
            continue
        mean_f1 = _stats.mean(per_f1)
        std_f1  = _stats.stdev(per_f1) if len(per_f1) > 1 else 0.0
        mean_p  = _stats.mean(per_p)
        mean_r  = _stats.mean(per_r)

        def _rates(key: str, _dim: str = dim) -> list[float]:
            rates = []
            for m in sample_results:
                if _dim not in m:
                    continue
                dm = m[_dim]
                tp_tot = (dm["tp_verified"] + dm["tp_halluc"] +
                          dm["tp_mismatch"] + dm["tp_no_check"])
                rates.append(dm.get(key, 0) / tp_tot if tp_tot else 0.0)
            return rates

        rv_rates  = _rates("tp_verified")
        rh_rates  = _rates("tp_halluc")
        rmm_rates = _rates("tp_mismatch")
        rnc_rates = _rates("tp_no_check")
        mean_rv  = _stats.mean(rv_rates)  if rv_rates  else 0.0
        mean_rh  = _stats.mean(rh_rates)  if rh_rates  else 0.0
        mean_rmm = _stats.mean(rmm_rates) if rmm_rates else 0.0
        mean_rnc = _stats.mean(rnc_rates) if rnc_rates else 0.0
        std_rv   = _stats.stdev(rv_rates)  if len(rv_rates)  > 1 else 0.0
        std_rh   = _stats.stdev(rh_rates)  if len(rh_rates)  > 1 else 0.0
        std_rmm  = _stats.stdev(rmm_rates) if len(rmm_rates) > 1 else 0.0
        std_rnc  = _stats.stdev(rnc_rates) if len(rnc_rates) > 1 else 0.0
        tp_v  = sum(m.get(dim, {}).get("tp_verified",  0) for m in sample_results)
        tp_h  = sum(m.get(dim, {}).get("tp_halluc",    0) for m in sample_results)
        tp_mm = sum(m.get(dim, {}).get("tp_mismatch",  0) for m in sample_results)
        tp_nc = sum(m.get(dim, {}).get("tp_no_check",  0) for m in sample_results)

        print(f"\n  [{dim}]")
        print(f"    Verified Precision : {mean_p:.3f}")
        print(f"    Verified Recall    : {mean_r:.3f}")
        print(f"    Verified F1        : {mean_f1:.3f} ± {std_f1:.3f}")
        print(f"    per-sample F1      : {[round(f, 3) for f in per_f1]}")
        print(f"    TP flow (mean ± std across {n_samp} samples):")
        print(f"      verified  : {mean_rv*100:5.1f}% ± {std_rv*100:.1f}%")
        print(f"      halluc    : {mean_rh*100:5.1f}% ± {std_rh*100:.1f}%")
        print(f"      mismatch  : {mean_rmm*100:5.1f}% ± {std_rmm*100:.1f}%")
        if tp_nc > 0:
            print(f"    TP counts (×{n_samp} samples): "
                  f"verified={tp_v}  halluc={tp_h}  mismatch={tp_mm}  no_check={tp_nc}")
            print(f"    ⚠ WARNING: {tp_nc} TP pairs were not checked by Stage 3. "
                  f"Re-run with --stages 2 3 4.")
        else:
            print(f"    TP counts (×{n_samp} samples): "
                  f"verified={tp_v}  halluc={tp_h}  mismatch={tp_mm}")

        all_dim_metrics[dim] = {
            "verified_precision_mean": round(mean_p, 4),
            "verified_recall_mean":    round(mean_r, 4),
            "verified_f1_mean":        round(mean_f1, 4),
            "verified_f1_std":         round(std_f1, 4),
            "verified_f1_per_sample":  [round(f, 4) for f in per_f1],
            "tp_flow_mean": {
                "verified":  round(mean_rv,  4), "halluc":   round(mean_rh,  4),
                "mismatch":  round(mean_rmm, 4), "no_check": round(mean_rnc, 4),
            },
            "tp_flow_std": {
                "verified":  round(std_rv,  4), "halluc":   round(std_rh,  4),
                "mismatch":  round(std_rmm, 4), "no_check": round(std_rnc, 4),
            },
            "tp_counts_total": {
                "tp_verified": tp_v, "tp_halluc": tp_h,
                "tp_mismatch": tp_mm, "tp_no_check": tp_nc,
            },
        }
    return {"dims": all_dim_metrics}


# main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results", nargs="+", required=True,
                    help="Grouping result JSON file(s) from run_grouping_bench.py")
    ap.add_argument("--input", required=True,
                    help="survey_grouping.json (for evidence lookup)")
    ap.add_argument("--mode", default="full",
                    choices=["abstract", "abstract+intro", "full"],
                    help="Content mode used during evaluation (default: full)")
    ap.add_argument("--output-dir", default="results/eval_pipeline",
                    help="Directory to save hallucinations_grouping.json and "
                         "reasoning_check_grouping.json")
    ap.add_argument("--base-url",  default=os.getenv("NOVGAUGE_BASE_URL"),
                    help="LLM API base URL (for Stage 3 judge)")
    ap.add_argument("--api-key",   default=os.getenv("NOVGAUGE_API_KEY"),
                    help="LLM API key (for Stage 3 judge)")
    ap.add_argument("--model",     default="openai/gpt-5.2",
                    help="Judge model for Stage 3 (default: openai/gpt-5.2)")
    ap.add_argument("--judge-temperature", type=float, default=0.0,
                    help="Judge model temperature (default: 0)")
    ap.add_argument("--judge-max-tokens",  type=int,   default=512,
                    help="Judge model max output tokens (default: 512)")
    ap.add_argument("--workers",   type=int, default=8)
    ap.add_argument("--stages",    nargs="+", type=int, default=[1, 2, 3, 4],
                    choices=[1, 2, 3, 4],
                    help="Which stages to run (default: 1 2 3 4)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip already-done Stage 3 records")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to summary.txt
    class _Tee:
        def __init__(self, f): self._f = f
        def write(self, s): sys.__stdout__.write(s); self._f.write(s)
        def flush(self): sys.__stdout__.flush(); self._f.flush()

    summary_path = output_dir / "summary.txt"
    _summary_file = open(summary_path, "w", encoding="utf-8")
    sys.stdout = _Tee(_summary_file)

    try:
        # Load records
        records = load_records(args.results)
        print(f"\nLoaded {len(records)} total records")
        print(f"  results : {args.results}")

        # Load item lookup for Stage 2
        items = load_json(args.input)
        item_lookup = _build_lookup(items)  # type: ignore[arg-type]
        print(f"  input   : {args.input}  ({len(item_lookup)} items)")

        # Filter records to only those present in the input dataset
        valid_item_ids = {it["item_id"] for it in items}  # type: ignore[index]
        before = len(records)
        records = [r for r in records if r.get("item_id") in valid_item_ids]
        if len(records) < before:
            print(f"  filtered: {before} → {len(records)} records "
                  f"(skipped {before - len(records)} not in input dataset)")

        all_metrics: dict = {
            "results": args.results,
            "mode": args.mode,
            "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        }

        # Stage 1
        if 1 in args.stages:
            s1 = stage1_accuracy(records)
            all_metrics["stage1"] = s1

        # Stage 2
        sample_indices = sorted(set(r.get("sample_idx", 0) for r in records))
        per_sample_hallucinated: list[set[tuple[str, str, int]]] = []
        if 2 in args.stages:
            per_sample_hallucinated, s2 = stage2_hallucination(
                records, item_lookup, args.mode, output_dir
            )
            all_metrics["stage2"] = s2
        elif 3 in args.stages:
            # Stage 2 skipped but Stage 3 needs per_sample_hallucinated — use empty sets
            per_sample_hallucinated = [set() for _ in sample_indices]

        # Stage 3
        if 3 in args.stages:
            cfg = LLMConfig(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
            )
            s3 = stage3_mismatch(
                records, per_sample_hallucinated, cfg,
                workers=args.workers,
                output_dir=output_dir,
                skip_existing=args.skip_existing,
                item_lookup=item_lookup if 2 in args.stages else None,
            )
            all_metrics["stage3"] = s3

        # Stage 4
        if 4 in args.stages and 2 in args.stages:
            s4 = stage4_verified_f1(records, per_sample_hallucinated, output_dir)
            all_metrics["stage4"] = s4

        # Save metrics.json
        metrics_path = output_dir / "metrics.json"
        save_json(all_metrics, metrics_path)
        print(f"\n  Metrics → {metrics_path}")
        print(f"  Summary → {summary_path}")
        print("\nDone.")
    finally:
        sys.stdout = sys.__stdout__
        _summary_file.close()


if __name__ == "__main__":
    main()
