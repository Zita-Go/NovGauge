"""Evaluate pairwise predictions through similarity, evidence, reasoning, and Verified F1 stages."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).parent
REPO = BASE.parent.parent
sys.path.insert(0, str(REPO))

from bench.evaluation.check_evidence import check_file as check_evidence_file, _build_lookup
from bench.evaluation.check_reasoning import build_prompt, call_llm, process_record
from bench.llm_client import LLMConfig

DIMS = ["task", "problem", "method"]


# helpers
def load_json(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:.1f}%" if denom else "n/a"


def _n_samples(records: list[dict]) -> int:
    return max((len(r.get("samples", [])) for r in records), default=1)


def _expand_for_sample(records: list[dict], s_idx: int) -> list[dict]:
    """Return virtual records where pred and llm_output come from samples[s_idx].

    Sets virtual["samples"] = [samples[s_idx]] so that check_evidence._get_llm_output
    and check_reasoning.process_record automatically read the correct sample.
    """
    out = []
    for r in records:
        samps = r.get("samples", [])
        if not samps:
            if s_idx == 0:
                out.append(r)
        elif s_idx < len(samps):
            out.append({**r, "pred": samps[s_idx]["pred"], "samples": [samps[s_idx]]})
    return out


# helpers for per-sample metrics
def _sample_metrics(dim_recs: list[dict], s_idx: int) -> dict:
    """Compute acc/prec/rec/f1 using samples[s_idx].pred for each record."""
    tp = fp = fn = tn = 0
    for r in dim_recs:
        samps = r.get("samples", [])
        p  = samps[s_idx]["pred"] if s_idx < len(samps) else r.get("pred")
        lb = r.get("label")
        if p is None or lb is None:
            continue
        if   lb == 1 and p == 1: tp += 1
        elif lb == 1 and p == 0: fn += 1
        elif lb == 0 and p == 1: fp += 1
        else:                    tn += 1
    n    = tp + fp + fn + tn
    acc  = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "acc": acc, "prec": prec, "rec": rec, "f1": f1}


# Stage 1: accuracy
def stage1_accuracy(records: list[dict]) -> dict:
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 1 — Similarity Judgment Accuracy")
    print("=" * 65)

    metrics: dict = {"n_samples": 1, "dims": {}}

    # Detect multi-sample: any record has >1 sample
    n_samples_max = max((len(r.get("samples", [])) for r in records), default=1)
    is_multi = n_samples_max > 1
    metrics["n_samples"] = n_samples_max

    for dim in DIMS:
        sub = [r for r in records if r.get("dim") == dim]
        if not sub:
            continue

        if is_multi:
            dim_recs = [r for r in sub if r.get("samples")]
            if not dim_recs:
                continue
            per_sample = [_sample_metrics(dim_recs, s_idx)
                          for s_idx in range(n_samples_max)]
            f1s  = [m["f1"]   for m in per_sample]
            accs = [m["acc"]  for m in per_sample]
            ps   = [m["prec"] for m in per_sample]
            rs   = [m["rec"]  for m in per_sample]

            mean_f1  = _stats.mean(f1s)
            std_f1   = _stats.stdev(f1s)  if len(f1s)  > 1 else 0.0
            mean_acc = _stats.mean(accs)
            mean_p   = _stats.mean(ps)
            mean_r   = _stats.mean(rs)

            print(f"\n  [{dim}]  n={len(dim_recs)}  n_samples={n_samples_max}")
            print(f"    Accuracy : {mean_acc:.3f}  (mean across samples)")
            print(f"    Precision: {mean_p:.3f}")
            print(f"    Recall   : {mean_r:.3f}")
            print(f"    F1       : {mean_f1:.3f} ± {std_f1:.3f}")
            print(f"    per-sample F1: {[round(f, 3) for f in f1s]}")
            metrics["dims"][dim] = {
                "n": len(dim_recs), "accuracy": round(mean_acc, 4),
                "precision": round(mean_p, 4), "recall": round(mean_r, 4),
                "f1_mean": round(mean_f1, 4), "f1_std": round(std_f1, 4),
                "f1_per_sample": [round(f, 4) for f in f1s],
            }
        else:
            valid = [r for r in sub if r.get("pred") is not None]
            if not valid:
                continue
            tp = sum(1 for r in valid if r["pred"] == 1 and r.get("label") == 1)
            fp = sum(1 for r in valid if r["pred"] == 1 and r.get("label") == 0)
            fn = sum(1 for r in valid if r["pred"] == 0 and r.get("label") == 1)
            tn = sum(1 for r in valid if r["pred"] == 0 and r.get("label") == 0)
            total = len(valid)

            acc  = pct(tp + tn, total)
            prec = pct(tp, tp + fp) if (tp + fp) else "n/a"
            rec  = pct(tp, tp + fn) if (tp + fn) else "n/a"
            f1_val = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0
            f1   = f"{f1_val:.3f}"

            print(f"\n  [{dim}]  n={total}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
            print(f"    Accuracy : {acc}")
            print(f"    Precision: {prec}")
            print(f"    Recall   : {rec}")
            print(f"    F1       : {f1}")
            metrics["dims"][dim] = {
                "n": total, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "accuracy": round((tp + tn) / total, 4) if total else 0,
                "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
                "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
                "f1": round(f1_val, 4),
            }

    return metrics


# Stage 2: hallucination
def _run_halluc_check(
    group: list[dict],
    group_label: str,
    lookup: dict[str, dict],
    mode: str,
    tmp_path: Path,
) -> tuple[dict, set[tuple[str, str]]]:
    """Run check_evidence on one group; return (stats, hallucinated_set)."""
    save_json(group, tmp_path)
    stats = check_evidence_file(tmp_path, lookup, mode)
    tmp_path.unlink(missing_ok=True)
    hallucinated: set[tuple[str, str]] = {
        (det["pair_id"], det["dim"]) for det in stats["details"]
    }
    return stats, hallucinated


def _print_halluc_group(group_name: str, group: list[dict], stats: dict) -> dict:
    n  = stats["has_evidence"] - stats["item_not_found"]
    ha = stats["hallucination"]
    pa = stats.get("paraphrase_both", 0)
    print(f"\n  {group_name}  (n={len(group)}, with evidence={stats['has_evidence']})")
    print(f"    Verbatim       : {stats['verbatim_both']:4d} / {n}  ({pct(stats['verbatim_both'], n)})")
    print(f"    Artifact-only  : {stats['artifact_both']:4d} / {n}  ({pct(stats['artifact_both'], n)})")
    print(f"    4-gram soft   : {pa:4d} / {n}  ({pct(pa, n)})")
    print(f"    Hallucination  : {ha:4d} / {n}  ({pct(ha, n)})")
    dim_ha: dict[str, int] = {}
    for det in stats["details"]:
        d = det.get("dim", "")
        dim_ha[d] = dim_ha.get(d, 0) + 1
    print(f"    {'dim':<10} {'hall':>5} / {'total':>5}   {'hall%':>6}")
    print(f"    {'-'*36}")
    dims_metrics: dict = {}
    for dim in DIMS:
        dim_n    = sum(1 for r in group if r.get("dim") == dim)
        dim_ha_n = dim_ha.get(dim, 0)
        print(f"    {dim:<10} {dim_ha_n:>5} / {dim_n:>5}   {pct(dim_ha_n, dim_n):>6}")
        if dim_n:
            dims_metrics[dim] = {"hallucination": dim_ha_n, "total": dim_n,
                                 "halluc_rate": round(dim_ha_n / dim_n * 100, 2)}
    return {
        "n": len(group), "with_evidence": n,
        "verbatim": stats["verbatim_both"], "artifact_only": stats["artifact_both"],
        "paraphrase": pa,
        "hallucination": ha,
        "halluc_rate": round(ha / n * 100, 2) if n else 0,
        "dims": dims_metrics,
    }


def stage2_hallucination(
    records: list[dict],
    lookup: dict[str, dict],
    mode: str,
    output_dir: Path,
) -> list[set[tuple[str, str]]]:
    """Check evidence quality per sample.

    Returns list of hallucinated (pair_id, dim) sets, one per sample.
    Single-sample: returns a list of one set.
    """
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 2 — Hallucination Rate  (pred=1 only)")
    print("=" * 65)

    n_samp = _n_samples(records)
    is_multi = n_samp > 1
    if is_multi:
        print(f"  [multi-sample] n_samples={n_samp}, computing per-sample rates")

    per_sample_hallucinated: list[set[tuple[str, str]]] = []
    all_details: list[dict] = []

    # Per-sample stats for mean±std
    tp_halluc_rates: list[float] = []
    fp_halluc_rates: list[float] = []

    for s_idx in range(n_samp):
        expanded  = _expand_for_sample(records, s_idx) if is_multi else records
        pred1     = [r for r in expanded if r.get("pred") == 1]
        tp_recs   = [r for r in pred1 if r.get("label") == 1]
        fp_recs   = [r for r in pred1 if r.get("label") == 0]

        if not pred1:
            per_sample_hallucinated.append(set())
            continue

        hallucinated_s: set[tuple[str, str]] = set()
        halluc_group_metrics: dict = {}

        if is_multi:
            print(f"\n  ── sample {s_idx}  (pred=1: {len(pred1)}, TP: {len(tp_recs)}, FP: {len(fp_recs)})")

        for grp_key, group, grp_label in [
            ("tp", tp_recs, "TP (pred=1, label=1)"),
            ("fp", fp_recs, "FP (pred=1, label=0)"),
        ]:
            if not group:
                if not is_multi:
                    print(f"\n  {grp_label}: no records")
                continue
            tmp = output_dir / f"_tmp_s{s_idx}_{grp_key}.json"
            stats, h_set = _run_halluc_check(group, grp_label, lookup, mode, tmp)
            hallucinated_s |= h_set
            all_details.extend(stats["details"])

            n_ev = stats["has_evidence"] - stats["item_not_found"]
            ha   = stats["hallucination"]
            rate = ha / n_ev if n_ev else 0.0
            if grp_key == "tp":
                tp_halluc_rates.append(rate)
            else:
                fp_halluc_rates.append(rate)

            if is_multi:
                print(f"    {grp_label}: halluc={ha}/{n_ev} ({pct(ha, n_ev)})")
            else:
                grp_m = _print_halluc_group(grp_label, group, stats)
                halluc_group_metrics[grp_key] = grp_m

        per_sample_hallucinated.append(hallucinated_s)

    halluc_metrics: dict = {}
    if is_multi and (tp_halluc_rates or fp_halluc_rates):
        print(f"\n  Mean±std across {n_samp} samples:")
        if tp_halluc_rates:
            m = _stats.mean(tp_halluc_rates) * 100
            s = (_stats.stdev(tp_halluc_rates) * 100) if len(tp_halluc_rates) > 1 else 0.0
            print(f"    TP halluc: {m:.1f}% ± {s:.1f}%")
            halluc_metrics["tp_halluc_mean"] = round(m, 2)
            halluc_metrics["tp_halluc_std"]  = round(s, 2)
            halluc_metrics["tp_halluc_per_sample"] = [round(r * 100, 2) for r in tp_halluc_rates]
        if fp_halluc_rates:
            m = _stats.mean(fp_halluc_rates) * 100
            s = (_stats.stdev(fp_halluc_rates) * 100) if len(fp_halluc_rates) > 1 else 0.0
            print(f"    FP halluc: {m:.1f}% ± {s:.1f}%")
            halluc_metrics["fp_halluc_mean"] = round(m, 2)
            halluc_metrics["fp_halluc_std"]  = round(s, 2)
            halluc_metrics["fp_halluc_per_sample"] = [round(r * 100, 2) for r in fp_halluc_rates]
    elif tp_halluc_rates or fp_halluc_rates:
        if tp_halluc_rates:
            halluc_metrics["tp_halluc_rate"] = round(tp_halluc_rates[0] * 100, 2)
        if fp_halluc_rates:
            halluc_metrics["fp_halluc_rate"] = round(fp_halluc_rates[0] * 100, 2)
        halluc_metrics.update(halluc_group_metrics)

    if all_details:
        save_json(all_details, output_dir / "hallucinations.json")
        print(f"\n  Hallucination details → {output_dir / 'hallucinations.json'}")

    all_paraphrase_details: list[dict] = []
    for s_idx in range(n_samp):
        expanded = _expand_for_sample(records, s_idx) if n_samp > 1 else records
        pred1    = [r for r in expanded if r.get("pred") == 1]
        for grp_key, group in [("tp", [r for r in pred1 if r.get("label") == 1]),
                                ("fp", [r for r in pred1 if r.get("label") == 0])]:
            if not group:
                continue
            tmp = output_dir / f"_tmp_para_s{s_idx}_{grp_key}.json"
            save_json(group, tmp)
            try:
                st = check_evidence_file(tmp, lookup, mode)
                for d in st.get("paraphrase_details", []):
                    d["sample_idx"] = s_idx
                    d["group"] = grp_key
                all_paraphrase_details.extend(st.get("paraphrase_details", []))
            finally:
                tmp.unlink(missing_ok=True)
    if all_paraphrase_details:
        save_json(all_paraphrase_details, output_dir / "paraphrases.json")
        print(f"  4-gram soft matches  → {output_dir / 'paraphrases.json'}")

    return per_sample_hallucinated, halluc_metrics


# Stage 3: mismatch rate
def stage3_mismatch(
    records: list[dict],
    per_sample_hallucinated: list[set[tuple[str, str]]],
    cfg: LLMConfig,
    workers: int,
    output_dir: Path,
    skip_existing: bool = True,
) -> None:
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 3 — Mismatch Rate  (pred=1, evidence verified)")
    print("=" * 65)

    n_samp   = len(per_sample_hallucinated)
    is_multi = n_samp > 1
    if is_multi:
        print(f"  [multi-sample] n_samples={n_samp}")

    out_path = output_dir / "reasoning_check.json"
    done: list[dict] = []
    existing: set[tuple] = set()
    if skip_existing and out_path.exists():
        done = load_json(out_path)
        # Only skip records where the judge actually returned a result (supported is not None)
        existing = {(d["pair_id"], d["dim"], d.get("sample_idx", 0))
                    for d in done if d.get("supported") is not None}
        done = [d for d in done if d.get("supported") is not None]
        print(f"  Skipping {len(existing)} already done")

    # Build to_run across all samples
    to_run: list[tuple[dict, int]] = []
    total_candidates = 0
    for s_idx, hallucinated_s in enumerate(per_sample_hallucinated):
        expanded = _expand_for_sample(records, s_idx) if is_multi else records
        for r in expanded:
            if r.get("pred") != 1:
                continue
            if (r.get("pair_id"), r.get("dim")) in hallucinated_s:
                continue
            total_candidates += 1
            if (r.get("pair_id"), r.get("dim"), s_idx) not in existing:
                to_run.append((r, s_idx))

    print(f"\n  Total candidates (all samples): {total_candidates}")
    print(f"  To run                        : {len(to_run)}")

    def _process(r: dict, s_idx: int) -> dict:
        result = process_record(r, cfg)
        result["sample_idx"] = s_idx
        return result

    if to_run:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, r, s_idx): (r, s_idx) for r, s_idx in to_run}
            for fut in as_completed(futures):
                done.append(fut.result())
                completed += 1
                if completed % 50 == 0 or completed == len(to_run):
                    print(f"    {completed}/{len(to_run)}  "
                          f"supported={sum(1 for d in done if d.get('supported') is True)}")
                    save_json(done, out_path)

    save_json(done, out_path)

    valid = [d for d in done if d.get("supported") is not None]

    mismatch_metrics: dict = {}
    if is_multi:
        # Per-sample mismatch rate (overall + TP/FP split), then mean±std
        sample_rates:    list[float] = []
        tp_sample_rates: list[float] = []
        fp_sample_rates: list[float] = []
        print(f"\n  Per-sample mismatch rates (overall):")
        print(f"  {'sample':>7}  {'checked':>8}  {'True':>6}  {'False':>6}  {'mismatch%':>10}  {'TP_mm%':>8}  {'FP_mm%':>8}")
        print(f"  {'-'*60}")
        for s_idx in range(n_samp):
            sub    = [d for d in valid if d.get("sample_idx", 0) == s_idx]
            tp_sub = [d for d in sub if d.get("label") == 1]
            fp_sub = [d for d in sub if d.get("label") == 0]
            if not sub:
                continue
            sup_f    = sum(1 for d in sub if not d["supported"])
            tp_sup_f = sum(1 for d in tp_sub if not d["supported"])
            fp_sup_f = sum(1 for d in fp_sub if not d["supported"])
            rate     = sup_f    / len(sub)    if sub    else 0.0
            tp_rate  = tp_sup_f / len(tp_sub) if tp_sub else 0.0
            fp_rate  = fp_sup_f / len(fp_sub) if fp_sub else 0.0
            sample_rates.append(rate)
            tp_sample_rates.append(tp_rate)
            fp_sample_rates.append(fp_rate)
            print(f"  {s_idx:>7}  {len(sub):>8}  {len(sub)-sup_f:>6}  {sup_f:>6}  {pct(sup_f, len(sub)):>10}  {pct(tp_sup_f, len(tp_sub)):>8}  {pct(fp_sup_f, len(fp_sub)):>8}")
        if sample_rates:
            m  = _stats.mean(sample_rates)    * 100
            s  = (_stats.stdev(sample_rates)    * 100) if len(sample_rates)    > 1 else 0.0
            tm = _stats.mean(tp_sample_rates)  * 100
            ts = (_stats.stdev(tp_sample_rates) * 100) if len(tp_sample_rates) > 1 else 0.0
            fm = _stats.mean(fp_sample_rates)  * 100
            fs = (_stats.stdev(fp_sample_rates) * 100) if len(fp_sample_rates) > 1 else 0.0
            print(f"\n  Mean±std mismatch (overall): {m:.1f}% ± {s:.1f}%")
            print(f"  Mean±std mismatch (TP only): {tm:.1f}% ± {ts:.1f}%")
            print(f"  Mean±std mismatch (FP only): {fm:.1f}% ± {fs:.1f}%")
            mismatch_metrics["mismatch_mean"]          = round(m,  2)
            mismatch_metrics["mismatch_std"]           = round(s,  2)
            mismatch_metrics["mismatch_per_sample"]    = [round(r * 100, 2) for r in sample_rates]
            mismatch_metrics["tp_mismatch_mean"]       = round(tm, 2)
            mismatch_metrics["tp_mismatch_std"]        = round(ts, 2)
            mismatch_metrics["tp_mismatch_per_sample"] = [round(r * 100, 2) for r in tp_sample_rates]
            mismatch_metrics["fp_mismatch_mean"]       = round(fm, 2)
            mismatch_metrics["fp_mismatch_std"]        = round(fs, 2)
            mismatch_metrics["fp_mismatch_per_sample"] = [round(r * 100, 2) for r in fp_sample_rates]
    else:
        # Single-sample: original detailed output
        print(f"\n  Results  (supported=True → reasoning verified, False → mismatch)")
        print(f"  {'dim':<10} {'checked':>8} {'True':>8} {'False':>8} {'mismatch%':>12}")
        print(f"  {'-'*50}")
        dims_mismatch: dict = {}
        for dim in DIMS:
            sub = [d for d in valid if d.get("dim") == dim]
            if not sub:
                continue
            sup_t = sum(1 for d in sub if d["supported"])
            sup_f = sum(1 for d in sub if not d["supported"])
            print(f"  {dim:<10} {len(sub):>8} {sup_t:>8} {sup_f:>8} {pct(sup_f, len(sub)):>12}")
            dims_mismatch[dim] = {"checked": len(sub), "supported": sup_t, "mismatch": sup_f,
                                  "mismatch_rate": round(sup_f / len(sub) * 100, 2) if sub else 0}
        if valid:
            sup_t = sum(1 for d in valid if d["supported"])
            sup_f = sum(1 for d in valid if not d["supported"])
            print(f"  {'ALL':<10} {len(valid):>8} {sup_t:>8} {sup_f:>8} {pct(sup_f, len(valid)):>12}")
            mismatch_metrics["dims"] = dims_mismatch
            mismatch_metrics["overall"] = {"checked": len(valid), "supported": sup_t, "mismatch": sup_f,
                                           "mismatch_rate": round(sup_f / len(valid) * 100, 2) if valid else 0}
        print(f"\n  TP / FP split:")
        print(f"  {'group':<20} {'checked':>8} {'True':>8} {'False':>8} {'mismatch%':>12}")
        print(f"  {'-'*60}")
        for label_val, label_name, key in [(1, "TP (label=1)", "tp"), (0, "FP (label=0)", "fp")]:
            sub = [d for d in valid if d.get("label") == label_val]
            if not sub:
                continue
            sup_t = sum(1 for d in sub if d["supported"])
            sup_f = sum(1 for d in sub if not d["supported"])
            print(f"  {label_name:<20} {len(sub):>8} {sup_t:>8} {sup_f:>8} {pct(sup_f, len(sub)):>12}")
            mismatch_metrics[f"{key}_mismatch_rate"] = round(sup_f / len(sub) * 100, 2) if sub else 0

    print(f"\n  Reasoning check results → {out_path}")
    return mismatch_metrics


# Stage 4: verified F1
def stage4_verified_f1(
    records: list[dict],
    per_sample_hallucinated: list[set[tuple[str, str]]],
    output_dir: Path,
) -> dict:
    """Compute per-dimension Verified F1 and the true-positive breakdown.
    Verification requires both evidence and reasoning checks to pass."""
    import statistics as _stats

    print("\n" + "=" * 65)
    print("  Stage 4 — Verified F1  (pred=1, not hallucinated, supported)")
    print("=" * 65)

    # Load Stage 3 reasoning results: (pair_id, dim, sample_idx) → supported
    reasoning_path = output_dir / "reasoning_check.json"
    reasoning_lookup: dict[tuple[str, str, int], bool | None] = {}
    if reasoning_path.exists():
        for d in load_json(reasoning_path):
            key = (d["pair_id"], d["dim"], d.get("sample_idx", 0))
            reasoning_lookup[key] = d.get("supported")

    n_samp   = len(per_sample_hallucinated)
    is_multi = n_samp > 1

    # Collect per-sample, per-dim metrics
    sample_results: list[dict[str, dict]] = []

    for s_idx in range(n_samp):
        hallucinated_s = per_sample_hallucinated[s_idx]
        expanded = _expand_for_sample(records, s_idx) if is_multi else records
        dim_m: dict[str, dict] = {}
        for dim in DIMS:
            sub = [r for r in expanded
                   if r.get("dim") == dim and r.get("pred") is not None
                   and r.get("label") is not None and r.get("samples")]
            if not sub:
                continue
            tp_verified = tp_halluc = tp_mismatch = tp_no_check = fp = fn = 0
            for r in sub:
                pid   = r.get("pair_id", "")
                label = r["label"]
                pred  = r["pred"]
                if label == 1 and pred == 1:
                    if (pid, dim) in hallucinated_s:
                        tp_halluc += 1
                    else:
                        sup = reasoning_lookup.get((pid, dim, s_idx))
                        if   sup is True:  tp_verified  += 1
                        elif sup is False: tp_mismatch  += 1
                        else:              tp_no_check  += 1
                elif label == 1 and pred == 0:
                    fn += 1
                elif label == 0 and pred == 1:
                    fp += 1
            tp_total = tp_verified + tp_halluc + tp_mismatch + tp_no_check
            v_p  = tp_verified / (tp_total + fp) if (tp_total + fp) else 0.0
            v_r  = tp_verified / (tp_total + fn) if (tp_total + fn) else 0.0
            v_f1 = 2 * v_p * v_r / (v_p + v_r) if (v_p + v_r) else 0.0
            dim_m[dim] = {
                "tp_verified": tp_verified, "tp_halluc": tp_halluc,
                "tp_mismatch": tp_mismatch, "tp_no_check": tp_no_check,
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
        # TP breakdown: per-sample rates, then mean±std
        def _rates(key: str) -> list[float]:
            rates = []
            for m in sample_results:
                if dim not in m:
                    continue
                dm = m[dim]
                tp_tot = dm["tp_verified"] + dm["tp_halluc"] + dm["tp_mismatch"] + dm["tp_no_check"]
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
        # raw counts for reference
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
            print(f"    ⚠ WARNING: {tp_nc} TPs were not checked by Stage 3 "
                  f"(Stage 3 incomplete). Re-run with --stages 2 3 4.")
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
                "tp_verified":  tp_v,  "tp_halluc":   tp_h,
                "tp_mismatch":  tp_mm, "tp_no_check": tp_nc,
            },
        }
    return {"dims": all_dim_metrics}


# main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pos-results", nargs="+", required=True,
                    help="Positives run_bench result JSON file(s)")
    ap.add_argument("--neg-results", nargs="+", default=[],
                    help="Negatives run_bench result JSON file(s)")
    ap.add_argument("--pos-inputs",  nargs="+", required=True,
                    help="Positives input JSON file(s) (for evidence lookup)")
    ap.add_argument("--neg-inputs",  nargs="+", default=[],
                    help="Negatives input JSON file(s)")
    ap.add_argument("--mode", default="full",
                    choices=["abstract", "abstract+intro", "full"])
    ap.add_argument("--output-dir",  default="results/eval_pipeline")
    ap.add_argument("--workers",     type=int, default=8)
    ap.add_argument("--base-url",    default=os.getenv("NOVGAUGE_BASE_URL"),
                    help="LLM API base URL (required for Stage 3)")
    ap.add_argument("--api-key",     default=os.getenv("NOVGAUGE_API_KEY"),
                    help="LLM API key (required for Stage 3; defaults to NOVGAUGE_API_KEY)")
    ap.add_argument("--model",            default="openai/gpt-5.2",
                    help="Judge model for Stage 3 (default: openai/gpt-5.2)")
    ap.add_argument("--judge-temperature", type=float, default=0.0,
                    help="Judge model temperature (default: 0)")
    ap.add_argument("--judge-max-tokens",  type=int,   default=512,
                    help="Judge model max output tokens (default: 512)")
    ap.add_argument("--stages",      nargs="+", type=int, default=[1, 2, 3, 4],
                    choices=[1, 2, 3, 4],
                    help="Which stages to run (default: 1 2 3 4)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip already-done Stage 3 records")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout → summary.txt
    class _Tee:
        def __init__(self, f): self._f = f
        def write(self, s): sys.__stdout__.write(s); self._f.write(s)
        def flush(self): sys.__stdout__.flush(); self._f.flush()

    summary_path = output_dir / "summary.txt"
    _summary_file = open(summary_path, "w", encoding="utf-8")
    sys.stdout = _Tee(_summary_file)

    # Load all results
    all_records: list[dict] = []
    for p in args.pos_results:
        all_records.extend(load_json(p))
    for p in args.neg_results:
        all_records.extend(load_json(p))

    print(f"\nLoaded {len(all_records)} total records")
    print(f"  pos_results : {args.pos_results}")
    print(f"  neg_results : {args.neg_results}")

    # Build lookup for evidence checking (pair_id → item)
    lookup: dict[str, dict] = {}
    for p in args.pos_inputs + args.neg_inputs:
        items = load_json(p)
        lookup.update(_build_lookup(items))
    print(f"  lookup size : {len(lookup)} items")

    import datetime
    all_metrics: dict = {
        "model": args.pos_results[0] if args.pos_results else "",
        "mode": args.mode,
        "n_records": len(all_records),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    # Stage 1
    if 1 in args.stages:
        s1 = stage1_accuracy(all_records)
        all_metrics["stage1"] = s1

    # Stage 2
    per_sample_hallucinated: list[set[tuple[str, str]]] = [set()]
    if 2 in args.stages:
        per_sample_hallucinated, s2 = stage2_hallucination(
            all_records, lookup, args.mode, output_dir
        )
        all_metrics["stage2"] = s2
    elif 3 in args.stages:
        n_samp = _n_samples(all_records)
        per_sample_hallucinated = [set() for _ in range(n_samp)]

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
            all_records, per_sample_hallucinated, cfg,
            workers=args.workers,
            output_dir=output_dir,
            skip_existing=args.skip_existing,
        )
        all_metrics["stage3"] = s3

    # Stage 4
    if 4 in args.stages and 2 in args.stages:
        s4 = stage4_verified_f1(all_records, per_sample_hallucinated, output_dir)
        all_metrics["stage4"] = s4

    metrics_path = output_dir / "metrics.json"
    save_json(all_metrics, metrics_path)
    print(f"\n  All metrics → {metrics_path}")
    print(f"  Summary     → {summary_path}")
    print("\nDone.")

    sys.stdout = sys.__stdout__
    _summary_file.close()


if __name__ == "__main__":
    main()
