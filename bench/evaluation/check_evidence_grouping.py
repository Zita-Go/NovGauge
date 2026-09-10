"""Check evidence for each paper in a predicted group against source text and titles."""
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

CACHE_PATH = REPO / "bench/output/cache/paper_content_cache_human.json"

# Content cache (lazy-loaded)
_content_cache: dict | None = None


def _load_cache() -> dict:
    global _content_cache
    if _content_cache is None:
        if CACHE_PATH.exists():
            with open(CACHE_PATH, encoding="utf-8") as f:
                _content_cache = json.load(f)
        else:
            _content_cache = {}
    return _content_cache


# normalisation helpers (copied from check_evidence.py)
_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    # typographic quotes / dashes
    "–": "-", "—": "-",
    "'": "'", "'": "'",
    "“": '"', "”": '"',
    "­": "",  # soft hyphen
})

# Markdown image tag: ![alt](url)  or  ![](url)
_IMAGE_RE   = re.compile(r"!\[.*?\]\(.*?\)", re.DOTALL)
# Inline italic/bold image captions: *(caption)*
_CAPTION_RE = re.compile(r"\*\(.*?\)\*", re.DOTALL)
# Superscript footnote numbers attached to words: "drift1 " → "drift "
_FOOTNOTE_RE = re.compile(r"(?<=[A-Za-z])\d+(?=[\s,\.;)])")

# Heading aliases for reference section detection
_REF_ALIASES = {
    "references", "bibliography", "works cited",
    "acknowledgements", "acknowledgments", "appendix",
}


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_references(md_text: str) -> str:
    """Remove everything from the References heading onward."""
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if not m:
            continue
        heading = _normalize_heading_text(m.group(2))
        if any(heading == alias or heading.startswith(alias + " ")
               for alias in _REF_ALIASES):
            return "\n".join(lines[:i]).strip()
    return md_text


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
    ev_grams    = [ev[i : i + n] for i in range(len(ev) - n + 1)]
    ct_gram_set = set(ct[i : i + n] for i in range(len(ct) - n + 1))
    return sum(1 for g in ev_grams if g in ct_gram_set) / len(ev_grams)


def _strip_artifacts(text: str) -> str:
    """Remove PDF/markdown artifacts from content before matching."""
    text = _base_normalise(text)
    text = _IMAGE_RE.sub(" ", text)
    text = _CAPTION_RE.sub(" ", text)
    text = _FOOTNOTE_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(\[])")
_MIN_SENT_LEN  = 20


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

    Returns (overall_classification, per_sentence_classifications).
    """
    if not evidence:
        return "ok", []

    norm_ct_base     = _base_normalise(content)
    norm_ct_artifact = _strip_artifacts(content)

    sents = _split_sentences(evidence)
    if not sents:
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


# Paper content extraction
def _get_paper_content(paper: dict, mode: str) -> str:
    """Extract paper content for evidence checking (no token truncation).

    The paper title is always prepended so that evidence strings copied from
    the <TITLE_X> prompt field are not flagged as hallucinations.
    """
    title   = (paper.get("title") or "").strip()
    content = _get_paper_content_body(paper, mode)
    return (title + "\n" + content) if title else content


def _get_paper_content_body(paper: dict, mode: str) -> str:
    abstract  = (paper.get("abstract") or "").strip()
    md_path   = paper.get("markdown_path", "")

    if mode == "abstract":
        return abstract

    if mode == "abstract+intro":
        if md_path:
            cache = _load_cache()
            entry = cache.get(md_path)
            if entry:
                cached = entry.get("abstract+intro") or entry.get("abstract") or abstract
                return cached
        # Fallback: just return abstract (intro extraction requires the markdown)
        if md_path:
            full_path = REPO / md_path
            if full_path.exists():
                md_text = full_path.read_text(encoding="utf-8", errors="replace")
                # Extract intro section
                lines = md_text.splitlines()
                _INTRO_ALIASES = {
                    "introduction", "intro", "background", "overview",
                    "motivation", "preliminaries", "preliminary",
                }
                intro_start = None
                for i, line in enumerate(lines):
                    m = re.match(r"^(#{1,3})\s+(.*)", line)
                    if not m:
                        continue
                    heading = _normalize_heading_text(m.group(2))
                    if any(heading == a or heading.startswith(a + " ")
                           or heading.startswith(a + ":") for a in _INTRO_ALIASES):
                        intro_start = i
                    elif intro_start is not None:
                        intro = "\n".join(lines[intro_start:i]).strip()
                        return f"{abstract}\n\n{intro}".strip() if intro else abstract
                if intro_start is not None:
                    intro = "\n".join(lines[intro_start:]).strip()
                    return f"{abstract}\n\n{intro}".strip() if intro else abstract
        return abstract

    # full
    if md_path:
        full_path = REPO / md_path
        if full_path.exists():
            md_text = full_path.read_text(encoding="utf-8", errors="replace")
            return strip_references(md_text)
    return abstract


# Build lookup
def _build_lookup(items: list[dict]) -> dict[str, dict]:
    """Build {item_id: item} lookup from survey_grouping items."""
    return {item["item_id"]: item for item in items if item.get("item_id")}


# Main check logic
def check_file(
    results_path: Path,
    item_lookup: dict[str, dict],
    mode: str,
) -> dict[str, Any]:
    """Check evidence quality for all predicted groups in a grouping result file.

    Returns a stats dict with keys:
      total_groups, verbatim, artifact, hallucination, details
    and also:
      hallucinated_set: set of (item_id, dim, group_idx) tuples
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # Support both raw list and wrapped {"results": [...]} format
    if isinstance(data, dict):
        records = data.get("results", [])
    else:
        records = data

    stats: dict[str, Any] = {
        "total_groups":  0,
        "verbatim":      0,
        "artifact":      0,
        "paraphrase":    0,
        "hallucination": 0,
        "details":       [],
    }
    hallucinated_set: set[tuple[str, str, int]] = set()

    for rec in records:
        item_id  = rec.get("item_id", "")
        dim      = rec.get("dim", "")
        rich_groups = rec.get("predicted_groups_rich") or []
        if not rich_groups:
            continue

        item = item_lookup.get(item_id)
        if item is None:
            continue

        # Build paper_index → paper dict for this item
        paper_by_idx: dict[int, dict] = {
            p["paper_index"]: p for p in item.get("papers", [])
        }

        for grp_idx, grp in enumerate(rich_groups):
            paper_indices = grp.get("paper_indices", [])
            evidence_map  = grp.get("evidence", {})
            if not isinstance(evidence_map, dict):
                evidence_map = {}
            reason = (grp.get("reason") or "").strip()

            group_cls_list: list[str] = []
            per_paper: list[dict] = []
            for paper_idx in paper_indices:
                ev_text = evidence_map.get(str(paper_idx), "")
                paper   = paper_by_idx.get(paper_idx, {})
                content = _get_paper_content(paper, mode)

                cls, _ = _classify_evidence(ev_text, content)
                if cls == "ok":
                    cls = "verbatim"
                group_cls_list.append(cls)

                per_paper.append({
                    "paper_index": paper_idx,
                    "title":       paper.get("title", ""),
                    "md_path":     paper.get("markdown_path", ""),
                    "cls":         cls,
                    "evidence":    ev_text,
                })

            # Classify the whole group
            if any(c == "hallucination" for c in group_cls_list):
                group_cls = "hallucination"
            elif any(c == "paraphrase" for c in group_cls_list):
                group_cls = "paraphrase"
            elif any(c == "artifact" for c in group_cls_list):
                group_cls = "artifact"
            else:
                group_cls = "verbatim"

            # One entry per group (not per paper) for human inspection
            stats["details"].append({
                "item_id":       item_id,
                "dim":           dim,
                "group_idx":     grp_idx,
                "sample_idx":    rec.get("sample_idx", 0),
                "paper_indices": paper_indices,
                "reason":        reason,
                "group_cls":     group_cls,
                "papers":        per_paper,
            })

            stats["total_groups"] += 1
            stats[group_cls] += 1

            if group_cls == "hallucination":
                hallucinated_set.add((item_id, dim, grp_idx))

    stats["hallucinated_set"] = hallucinated_set
    return stats


# CLI
def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:5.1f}%" if denom else "  n/a "


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results", required=True,
                    help="Grouping result JSON from run_grouping_bench.py")
    ap.add_argument("--input",   required=True,
                    help="survey_grouping.json input file")
    ap.add_argument("--mode",    default="full",
                    choices=["abstract", "abstract+intro", "full"])
    ap.add_argument("--save-details", metavar="PATH",
                    help="Save per-paper evidence details to a JSON file")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)
    item_lookup = _build_lookup(items)
    print(f"Loaded {len(item_lookup)} items from {args.input}")

    results_path = Path(args.results)
    print(f"Checking {results_path}  (mode={args.mode})")

    stats = check_file(results_path, item_lookup, args.mode)
    n = stats["total_groups"]

    print(f"\n{'='*65}")
    print(f"  Evidence Quality — Grouping  ({results_path.stem})")
    print(f"{'='*65}")
    print(f"  Total predicted groups: {n}")
    if n:
        print(f"  Verbatim      : {stats['verbatim']:4d} / {n}  "
              f"({_pct(stats['verbatim'], n)})")
        print(f"  Artifact-only : {stats['artifact']:4d} / {n}  "
              f"({_pct(stats['artifact'], n)})")
        print(f"  Char-ngram    : {stats['paraphrase']:4d} / {n}  "
              f"({_pct(stats['paraphrase'], n)})")
        print(f"  Hallucination : {stats['hallucination']:4d} / {n}  "
              f"({_pct(stats['hallucination'], n)})")

    if args.save_details and stats["details"]:
        out = Path(args.save_details)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats["details"], f, ensure_ascii=False, indent=2)
        print(f"\nDetails saved to: {out}")


if __name__ == "__main__":
    main()
