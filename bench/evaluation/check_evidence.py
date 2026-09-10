"""Check pairwise evidence against source paper text and titles."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

BASE = Path(__file__).parent
REPO = BASE.parent.parent
sys.path.insert(0, str(REPO))

from bench.evaluation.run_bench import get_content


# normalisation helpers
_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    # typographic quotes / dashes
    "–": "-", "—": "-",
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "­": "",   # soft hyphen
})

# Markdown image tag: ![alt](url)  or  ![](url)
_IMAGE_RE   = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)
# Inline italic/bold image captions: *(caption)*
_CAPTION_RE = re.compile(r"\*\(.*?\)\*", re.DOTALL)
# Superscript footnote numbers attached to words: "drift1 " → "drift "
_FOOTNOTE_RE = re.compile(r"(?<=[A-Za-z])\d+(?=[\s,\.;)])")


SOFT_MATCH_THRESHOLD = 0.75   # character 4-gram recall threshold
_CHAR_NGRAM_N = 4

# Trailing ellipsis artifact: "sentence text. ... repeated phrase."
_TRAILING_ELLIPSIS_RE = re.compile(r'\s+\.{2,}\s+\S.*$', re.DOTALL)


def _base_normalise(text: str) -> str:
    """Unicode / ligature normalisation + whitespace collapse + lowercase."""
    text = text.translate(_LIGATURES)
    text = re.sub(r"-\s+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _strip_trailing_ellipsis(sent: str) -> str:
    """Remove trailing '... repeated text' model artifacts from evidence sentences."""
    cleaned = _TRAILING_ELLIPSIS_RE.sub("", sent).strip()
    return cleaned if cleaned else sent


def _char_ngram_recall(norm_sent: str, norm_content: str) -> float:
    """Character n-gram recall after removing all whitespace.

    Removes spaces from both sides before computing n-grams, so broken words
    like 'per forms' and 'performs' produce identical n-gram sets.
    Returns fraction of evidence n-grams found anywhere in content.
    """
    n  = _CHAR_NGRAM_N
    ev = re.sub(r"\s+", "", norm_sent)
    ct = re.sub(r"\s+", "", norm_content)
    if len(ev) < n:
        return 1.0 if ev in ct else 0.0
    ev_grams   = [ev[i : i + n] for i in range(len(ev) - n + 1)]
    ct_gram_set = set(ct[i : i + n] for i in range(len(ct) - n + 1))
    return sum(1 for g in ev_grams if g in ct_gram_set) / len(ev_grams)


def _strip_artifacts(text: str) -> str:
    """Remove PDF/markdown artifacts from content before matching.

    Applied to the *content* only.  Returns base-normalised text with
    image tags, figure captions, and footnote numbers removed.
    """
    text = _base_normalise(text)
    text = _IMAGE_RE.sub(" ", text)
    text = _CAPTION_RE.sub(" ", text)
    text = _FOOTNOTE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(\[])")
_MIN_SENT_LEN  = 20   # ignore fragments shorter than this


def _split_sentences(text: str) -> list[str]:
    """Split evidence into individual sentences for per-sentence checking."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if len(s.strip()) >= _MIN_SENT_LEN]


def _sent_in_content(sent: str, norm_ct_base: str, norm_ct_artifact: str) -> str:
    """Return 'verbatim' | 'artifact' | 'paraphrase' | 'hallucination' for a single sentence."""
    norm_s = _base_normalise(sent)
    if not norm_s:
        return "verbatim"
    if norm_s in norm_ct_base:
        return "verbatim"
    if norm_s in norm_ct_artifact:
        return "artifact"
    # No-whitespace fallback for broken-word artifacts (≥20 chars)
    if len(norm_s) >= 20:
        ns = re.sub(r"\s+", "", norm_s)
        nc = re.sub(r"\s+", "", norm_ct_artifact)
        if ns in nc:
            return "artifact"
    # Char n-gram soft match (catches PDF noise / broken words / minor paraphrase)
    if _char_ngram_recall(norm_s, norm_ct_artifact) >= SOFT_MATCH_THRESHOLD:
        return "paraphrase"
    return "hallucination"


def _classify_evidence(evidence: str, content: str) -> tuple[str, list[str]]:
    """Classify evidence as 'verbatim' | 'artifact' | 'hallucination'.

    Splits evidence into sentences and checks each individually.
    Returns (overall_classification, per_sentence_classifications).
    The overall classification is the worst across all sentences:
      verbatim < artifact < hallucination
    """
    if not evidence:
        return "ok", []

    norm_ct_base     = _base_normalise(content)
    norm_ct_artifact = _strip_artifacts(content)

    sents = _split_sentences(evidence)
    if not sents:
        # Single short fragment — check as whole
        sents = [evidence]
    sents = [_strip_trailing_ellipsis(s) for s in sents]

    per_sent = [_sent_in_content(s, norm_ct_base, norm_ct_artifact) for s in sents]

    if any(c == "hallucination" for c in per_sent):
        overall = "hallucination"
    elif any(c == "paraphrase" for c in per_sent):
        overall = "paraphrase"
    elif any(c == "artifact" for c in per_sent):
        overall = "artifact"
    else:
        overall = "verbatim"

    return overall, per_sent


# extract evidence from a result record
def _get_llm_output(record: dict) -> dict:
    """Support both old format (llm_output at top) and new format (samples[])."""
    if "samples" in record:
        samples = record["samples"]
        if samples:
            return samples[0].get("llm_output") or {}
        return {}
    return record.get("llm_output") or {}


# build pair-id → item lookup
def _build_lookup(items: list[dict]) -> dict[str, dict]:
    return {item["pair_id"]: item for item in items if item.get("pair_id")}


# main check logic
def check_file(
    results_path: Path,
    lookup: dict[str, dict],
    mode: str,
    verbose: bool = False,
) -> dict[str, Any]:
    with open(results_path, encoding="utf-8") as f:
        records = json.load(f)

    stats: dict[str, Any] = {
        "total":            0,
        "has_evidence":     0,
        "item_not_found":   0,
        "verbatim_both":    0,   # strict match on both A and B
        "artifact_both":    0,   # matches only after artifact stripping
        "paraphrase_both":    0,   # 4-gram soft match (PDF noise / minor paraphrase)
        "hallucination":      0,   # fails all checks
        "details":            [],  # hallucination records
        "paraphrase_details": [],  # 4-gram soft-match records for threshold inspection
    }

    for rec in records:
        lo = _get_llm_output(rec)
        ev_a_raw = lo.get("evidence_a") or ""
        ev_b_raw = lo.get("evidence_b") or ""
        ev_a = ev_a_raw.strip() if isinstance(ev_a_raw, str) else ""
        ev_b = ev_b_raw.strip() if isinstance(ev_b_raw, str) else ""

        stats["total"] += 1
        if not ev_a and not ev_b:
            continue
        stats["has_evidence"] += 1

        pid  = rec.get("pair_id", "")
        item = lookup.get(pid)
        if item is None:
            stats["item_not_found"] += 1
            continue

        paper_a = item.get("submitted_paper") or item.get("paper_a") or {}
        paper_b = item.get("prior_work")       or item.get("paper_b") or {}
        _, content_a = get_content(paper_a, mode, max_tokens=999_999)
        _, content_b = get_content(paper_b, mode, max_tokens=999_999)
        # Prepend the paper title so evidence that copies the title is not flagged
        # as hallucination (the LLM saw the title in the prompt via <TITLE_A/B>).
        title_a = (paper_a.get("title") or "").strip()
        title_b = (paper_b.get("title") or "").strip()
        if title_a:
            content_a = title_a + "\n" + content_a
        if title_b:
            content_b = title_b + "\n" + content_b

        cls_a, sents_a = _classify_evidence(ev_a, content_a)
        cls_b, sents_b = _classify_evidence(ev_b, content_b)

        # Both verbatim
        if cls_a in ("verbatim", "ok") and cls_b in ("verbatim", "ok"):
            stats["verbatim_both"] += 1
        # At least one hallucination
        elif "hallucination" in (cls_a, cls_b):
            pass  # handled below
        # At least one paraphrase (4-gram soft match), none hallucinated
        elif "paraphrase" in (cls_a, cls_b):
            stats["paraphrase_both"] += 1
        # Artifact only
        else:
            stats["artifact_both"] += 1

        entry = {
            "pair_id":   pid,
            "dim":       rec.get("dim"),
            "label":     rec.get("label"),
            "pred":      rec.get("pred"),
            "cls_a":     cls_a,
            "cls_b":     cls_b,
            "sents_a":   sents_a,
            "sents_b":   sents_b,
            "title_a":   paper_a.get("title", "")[:80],
            "title_b":   paper_b.get("title", "")[:80],
            "md_path_a": paper_a.get("markdown_path", ""),
            "md_path_b": paper_b.get("markdown_path", ""),
            "evidence_a": ev_a,
            "evidence_b": ev_b,
        }
        if "hallucination" in (cls_a, cls_b):
            stats["hallucination"] += 1
            stats["details"].append(entry)
        elif "paraphrase" in (cls_a, cls_b):
            stats["paraphrase_details"].append(entry)
        if "hallucination" in (cls_a, cls_b) and verbose:
                flags = f"A={cls_a} B={cls_b}"
                print(f"  HALLUCINATION [{rec.get('dim')}] {flags}")
                print(f"    {pid[:70]}")
                if cls_a == "hallucination":
                    print(f"    ev_a: '{ev_a[:80]}'")
                if cls_b == "hallucination":
                    print(f"    ev_b: '{ev_b[:80]}'")

    return stats


# pretty print
def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:5.1f}%" if denom else "  n/a "


def print_stats(label: str, stats: dict) -> None:
    n   = stats["has_evidence"]
    tot = stats["total"]
    v   = stats["verbatim_both"]
    ar  = stats["artifact_both"]
    ha  = stats["hallucination"]
    nf  = stats["item_not_found"]

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Total records           : {tot}")
    print(f"  Has evidence fields     : {n}  ({_pct(n, tot)})")
    if nf:
        print(f"  Item not found in input : {nf}  (excluded from rates)")
    evaluated = n - nf
    if evaluated == 0:
        print("  (nothing to evaluate)")
        return
    pa = stats["paraphrase_both"]
    print(f"  ── Evidence quality (per record, both sides) ──")
    print(f"  Verbatim match          : {v:4d} / {evaluated}  ({_pct(v, evaluated)})  "
          f"← strict substring")
    print(f"  PDF artifact only       : {ar:4d} / {evaluated}  ({_pct(ar, evaluated)})  "
          f"← passes after artifact stripping")
    print(f"  Char-ngram soft match   : {pa:4d} / {evaluated}  ({_pct(pa, evaluated)})  "
          f"← char 4-gram recall ≥ {SOFT_MATCH_THRESHOLD}")
    print(f"  Hallucination           : {ha:4d} / {evaluated}  ({_pct(ha, evaluated)})  "
          f"← fails all checks")
    print(f"  ── Rates ──")
    print(f"  Verbatim rate           : {_pct(v,               evaluated)}")
    print(f"  Soft-corrected rate     : {_pct(v + ar + pa,     evaluated)}")
    print(f"  Hallucination rate      : {_pct(ha,              evaluated)}")
    if stats["details"]:
        print(f"\n  First 3 hallucination examples:")
        for m in stats["details"][:3]:
            print(f"    [{m['cls_a']}/{m['cls_b']}] dim={m['dim']} "
                  f"label={m['label']} pred={m['pred']}")
            if m["cls_a"] == "hallucination":
                print(f"      A: {m['title_a']}")
                print(f"         ev: \"{m['evidence_a'][:100]}\"")
            if m["cls_b"] == "hallucination":
                print(f"      B: {m['title_b']}")
                print(f"         ev: \"{m['evidence_b'][:100]}\"")


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="+", required=True,
                   help="Result JSON file(s) to check")
    p.add_argument("--inputs",  nargs="+", required=True,
                   help="Corresponding input JSON file(s)")
    p.add_argument("--mode", default="full",
                   choices=["abstract", "abstract+intro", "full"],
                   help="Content mode used during evaluation (default: full)")
    p.add_argument("--verbose", action="store_true",
                   help="Print each hallucination as it is found")
    p.add_argument("--save-mismatches", metavar="PATH",
                   help="Save hallucination details to a JSON file")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_paths = [Path(p) for p in args.results]
    input_paths   = [Path(p) for p in args.inputs]
    if len(input_paths) == 1 and len(results_paths) > 1:
        input_paths = input_paths * len(results_paths)
    if len(results_paths) != len(input_paths):
        sys.exit("ERROR: --results and --inputs must have the same number of entries")

    all_details: list[dict] = []

    for res_path, inp_path in zip(results_paths, input_paths):
        print(f"\nLoading input : {inp_path}")
        with open(inp_path, encoding="utf-8") as f:
            items = json.load(f)
        lookup = _build_lookup(items)

        print(f"Checking      : {res_path}  (mode={args.mode})")
        stats = check_file(res_path, lookup, args.mode, verbose=args.verbose)
        label = f"{res_path.parent.name}/{res_path.stem}"
        print_stats(label, stats)
        all_details.extend(stats["details"])

    if args.save_mismatches and all_details:
        out = Path(args.save_mismatches)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_details, f, ensure_ascii=False, indent=2)
        print(f"\nHallucination details saved to: {out}")

    total_ha = len(all_details)
    if total_ha:
        print(f"\n{'='*65}")
        print(f"  Total hallucinations across all files: {total_ha}")
    else:
        print(f"\n  All evidence verified (verbatim or artifact-corrected).")


if __name__ == "__main__":
    main()
