"""
Multi-paper grouping benchmark for survey_grouping.json.

For each item (sentence with N cited papers), ask the model to identify
which papers belong to the same group in each dimension (task / problem / method).
Three separate LLM calls per item (one per dimension).

Evaluation (pairwise F1):
  - Only evaluated on dimensions with non-empty ground-truth groups (eval_dims)
  - Convert predicted and ground-truth groups to pairwise labels
  - Compute Precision / Recall / F1 per dimension

Usage:
  python bench/run_grouping_bench.py \\
    --input data/grouping.json \\
    --models openai/gpt-4o \\
    --content-modes abstract abstract+intro full \\
    --dims task problem method \\
    --output-dir results/grouping_test
"""
from __future__ import annotations

import argparse
import os
import json
import re
import time
import concurrent.futures
from pathlib import Path
from typing import Any

from openai import OpenAI

BASE       = Path(__file__).parent
REPO       = BASE.parent
PROMPT_DIR = BASE / "prompts"
CACHE_PATH = REPO / "bench/output/cache/paper_content_cache_human.json"

PROMPT_FILES = {
    "task":    PROMPT_DIR / "grouping_task_prompt.txt",
    "problem": PROMPT_DIR / "grouping_problem_prompt.txt",
    "method":  PROMPT_DIR / "grouping_method_prompt.txt",
}

DEFAULT_BASE_URL    = os.getenv("NOVGAUGE_BASE_URL")
DEFAULT_API_KEY     = os.getenv("NOVGAUGE_API_KEY")
DEFAULT_WORKERS     = 8
DEFAULT_TIMEOUT     = 120
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS  = 1024

DEFAULT_MAX_ABSTRACT_TOKENS  = 800
DEFAULT_MAX_INTRO_TOKENS     = 3000
DEFAULT_MAX_FULL_TOKENS      = 12000

# ── Tokenizer ─────────────────────────────────────────────────────────────────

_tokenizer: Any = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer = False
    return None if _tokenizer is False else _tokenizer


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    enc = _get_tokenizer()
    if enc is None:
        return text[: max_tokens * 4]
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])


# ── Content cache (lazy-loaded) ──────────────────────────────────────────────

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


# ── Markdown content extraction ───────────────────────────────────────────────

def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


_INTRO_ALIASES = {
    "introduction", "intro", "background", "overview",
    "motivation", "preliminaries", "preliminary",
}
_REF_ALIASES = {
    "references", "bibliography", "works cited",
    "acknowledgements", "acknowledgments", "appendix",
}


def extract_intro(md_text: str) -> str:
    lines = md_text.splitlines()
    intro_start = None
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if not m:
            continue
        heading = _normalize_heading_text(m.group(2))
        if any(heading == alias or heading.startswith(alias + " ") or heading.startswith(alias + ":") for alias in _INTRO_ALIASES):
            intro_start = i
        elif intro_start is not None:
            return "\n".join(lines[intro_start:i]).strip()
    if intro_start is not None:
        return "\n".join(lines[intro_start:]).strip()
    return ""


def strip_references(md_text: str) -> str:
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if not m:
            continue
        heading = _normalize_heading_text(m.group(2))
        if any(heading == alias or heading.startswith(alias + " ") for alias in _REF_ALIASES):
            return "\n".join(lines[:i]).strip()
    return md_text


def get_paper_content(paper: dict, mode: str, max_tokens: int) -> tuple[str, str]:
    """Return (content_type, content_text) for a paper.

    Loads from pre-extracted cache when available; falls back to runtime extraction.
    """
    abstract = (paper.get("abstract") or "").strip()
    md_path  = paper.get("markdown_path")

    if mode == "abstract":
        return "abstract", truncate_by_tokens(abstract, max_tokens)

    # abstract+intro: use pre-extracted cache; full: read markdown directly
    if md_path:
        full_path = REPO / md_path
        md_text = full_path.read_text(encoding="utf-8", errors="replace") if full_path.exists() else ""
    else:
        md_text = ""

    if mode == "abstract+intro":
        if md_path:
            cache = _load_cache()
            entry = cache.get(md_path)
            if entry:
                text = entry.get("abstract+intro") or abstract
                return "abstract+intro", truncate_by_tokens(text, max_tokens)
        # Fallback: runtime extraction
        intro = extract_intro(md_text) if md_text else ""
        text = f"{abstract}\n\n{intro}".strip() if intro else abstract
        return "abstract+intro", truncate_by_tokens(text, max_tokens)

    # full
    text = strip_references(md_text) if md_text else abstract
    return "full", truncate_by_tokens(text, max_tokens)


# ── Prompt building ────────────────────────────────────────────────────────────

def format_paper_block(paper: dict, content_type: str, content: str) -> str:
    title   = paper.get("title") or "(Unknown title)"
    authors = ", ".join(paper.get("authors") or []) or "Unknown"
    idx     = paper["paper_index"]
    lines   = [
        f"[Paper {idx}]",
        f"Title: {title}",
        f"Authors: {authors}",
        f"Content type: {content_type}",
        "",
    ]
    if content.strip():
        lines.append(content.strip())
    return "\n".join(lines)


def build_prompt(template: str, papers: list[dict], mode: str, max_tokens_per_paper: int) -> str:
    paper_blocks = []
    for p in papers:
        ct, content = get_paper_content(p, mode, max_tokens_per_paper)
        paper_blocks.append(format_paper_block(p, ct, content))

    papers_text = "\n\n---\n\n".join(paper_blocks)
    return (
        template
        .replace("<N_PAPERS>", str(len(papers)))
        .replace("<PAPERS>", papers_text)
    )


# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_groups(text: str, n_papers: int) -> tuple[list[list[int]], list[dict]] | tuple[None, None]:
    """Parse model output.

    Returns (index_groups, rich_groups) where:
      - index_groups: list of lists of 1-based paper indices (for eval)
      - rich_groups:  raw group objects with paper_indices/evidence/reason (for logging)
    Returns (None, None) on parse failure.
    """
    obj = _extract_json(text)
    if obj is None:
        return None, None

    raw_groups = obj.get("groups", [])
    if not isinstance(raw_groups, list):
        return None, None

    index_groups, rich_groups = _validate_groups(raw_groups, n_papers)
    return index_groups, rich_groups


def _fix_json_quotes(text: str) -> str:
    """Best-effort fix for unescaped double quotes inside JSON string values.

    Targets the pattern where model output contains literal "word" inside a
    string, e.g. `"(i.e., "layers")"`, by replacing inner quotes with single
    quotes.  Only applied when standard parsing fails.
    """
    import re as _re
    # Replace unescaped " inside JSON string values with '
    # Strategy: within each string token (between outer "..."), replace inner
    # unescaped " with '
    def _fix_string(m: re.Match) -> str:
        inner = m.group(1)
        # Replace unescaped " (not preceded by \) with '
        fixed = _re.sub(r'(?<!\\)"', "'", inner)
        return f'"{fixed}"'
    return _re.sub(r'"((?:[^"\\]|\\.)*)"', _fix_string, text)


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from model output.

    When multiple JSON blocks exist (e.g. model revises its answer), take the
    last valid one — the model's final answer supersedes earlier drafts.
    Falls back to quote-fixing when strict JSON parsing fails.
    """
    def _try_parse(s: str) -> dict | None:
        try:
            obj = json.loads(s)
            return obj if "groups" in obj else None
        except json.JSONDecodeError:
            pass
        # Retry with quote fix
        try:
            obj = json.loads(_fix_json_quotes(s))
            return obj if "groups" in obj else None
        except json.JSONDecodeError:
            return None

    # Try all code blocks, keep the last valid one
    last_obj: dict | None = None
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        obj = _try_parse(m.group(1))
        if obj is not None:
            last_obj = obj
    if last_obj is not None:
        return last_obj

    # Fallback: scan all top-level JSON objects containing "groups", take last
    for m in re.finditer(r'\{', text):
        start = m.start()
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start: i + 1]
                    obj = _try_parse(candidate)
                    if obj is not None:
                        last_obj = obj
                    break
    return last_obj


def _validate_groups(raw_groups: list, n_papers: int) -> tuple[list[list[int]], list[dict]]:
    """Validate and normalise groups from model output.

    Each element of raw_groups can be:
      - a list of ints  (old simple format)
      - a dict with "paper_indices" key (new rich format with evidence/reason)
    """
    seen: set[int] = set()
    index_groups: list[list[int]] = []
    rich_groups:  list[dict]      = []

    for grp in raw_groups:
        # Extract indices
        if isinstance(grp, list):
            raw_indices = grp
            evidence    = {}
            reason      = ""
        elif isinstance(grp, dict):
            raw_indices = grp.get("paper_indices", [])
            evidence    = grp.get("evidence", {})
            reason      = grp.get("reason", "")
        else:
            continue

        clean = []
        for idx in raw_indices:
            if isinstance(idx, int) and 1 <= idx <= n_papers and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        if len(clean) < 2:
            continue

        clean = sorted(clean)
        index_groups.append(clean)
        rich_groups.append({
            "paper_indices": clean,
            "evidence":      evidence,
            "reason":        reason,
        })

    return index_groups, rich_groups


# ── Evaluation ─────────────────────────────────────────────────────────────────

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
    fn = len(gt_pairs - pred_pairs)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ── LLM call ──────────────────────────────────────────────────────────────────

import re as _re_tb

_LOCAL_QWEN3_RE = _re_tb.compile(r"^qwen3(\.\d+)?-\d+b$", _re_tb.IGNORECASE)

_STREAMING_MODELS: set[str] = {"anthropic/claude-opus-4.7"}
_MODEL_API_NAME_MAP: dict[str, str] = {"anthropic/claude-opus-4.7": "claude-opus-4-7"}


def _thinking_extra_body(model: str, thinking_budget: int) -> dict | None:
    """Return extra_body for the configured maximum reasoning level, matching run_bench.py."""
    m = model.lower()

    # :thinking variants encode reasoning settings in the model name; omit extra_body.
    if m.endswith("-thinking") or m.endswith(":thinking"):
        return None

    if "deepseek" in m:
        if thinking_budget <= 0:
            return {"thinking": {"type": "disabled"}}
        return {"reasoning_effort": "max"}

    if thinking_budget <= 0:
        return None

    if _LOCAL_QWEN3_RE.match(model):
        return {"chat_template_kwargs": {"enable_thinking": True}}
    if "gpt-5" in m:
        return {"reasoning_effort": "xhigh"}
    if any(x in m for x in ["o1", "o3", "o4"]):
        return {"reasoning_effort": "high"}
    if "gemini-3" in m or "gemini3" in m:
        return {"generation_config": {"thinking_config": {"thinking_level": "high"}}}
    if "kimi" in m or "moonshot" in m:
        return {"thinking": {"type": "enabled"}}
    if "glm" in m:
        return {"thinking": {"type": "enabled"}}
    if "minimax" in m:
        return {"reasoning_split": True}
    if "qwen" in m:
        return {"enable_thinking": True}
    if "claude-opus-4.7" in m or "claude-opus-4-7" in m:
        return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "max"}}
    if "claude" in m:
        return {"thinking": {"type": "adaptive"}}
    return {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}


def _extract_reasoning(msg) -> str:
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not reasoning:
        import re as _re
        m = _re.search(r"<think>(.*?)</think>", msg.content or "", _re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
    return reasoning


def _call_llm_streaming(
    client: OpenAI, api_model: str, prompt: str,
    timeout: int, max_tokens: int, extra_body: dict | None,
) -> tuple[str, str | None, str]:
    """Stream the response and concatenate content and reasoning_content deltas.
    Set temperature=1 for Claude reasoning mode.
    """
    kwargs: dict = dict(
        model=api_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=max_tokens,
        timeout=timeout,
        stream=True,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content_parts.append(delta.content)
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            reasoning_parts.append(rc)
    return "".join(content_parts), None, "".join(reasoning_parts)


def call_llm(
    client: OpenAI, model: str, prompt: str,
    timeout: int, temperature: float, max_tokens: int,
    thinking_budget: int = 0,
) -> tuple[str, str | None, str]:
    """Return (raw_text, error_str, reasoning_content)."""
    api_model  = _MODEL_API_NAME_MAP.get(model, model)
    use_stream = model in _STREAMING_MODELS
    extra      = _thinking_extra_body(model, thinking_budget)
    try:
        if use_stream:
            raw, err, reasoning = _call_llm_streaming(
                client, api_model, prompt, timeout, max_tokens, extra,
            )
            if err:
                raise Exception(err)
        else:
            kwargs: dict = dict(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if extra:
                kwargs["extra_body"] = extra
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            reasoning = _extract_reasoning(msg)
            raw = msg.content or ""
        import re as _re
        raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
        return raw, None, reasoning
    except Exception as e:
        return "", str(e), ""


# ── Per-task worker ────────────────────────────────────────────────────────────

def process_one(
    item: dict, dim: str, template: str, mode: str,
    client: OpenAI, model: str,
    timeout: int, temperature: float, max_tokens: int,
    max_tokens_per_paper: int,
    sample_idx: int = 0,
    thinking_budget: int = 0,
    max_retries: int = 3,
) -> dict:
    prompt = build_prompt(template, item["papers"], mode, max_tokens_per_paper)
    n = item["n_papers"]

    import time as _time
    raw, error, reasoning = "", None, ""
    idx_groups, rich_groups = None, None
    for attempt in range(1, max_retries + 1):
        raw, error, reasoning = call_llm(
            client, model, prompt, timeout, temperature, max_tokens,
            thinking_budget=thinking_budget,
        )
        if not error:
            idx_groups, rich_groups = parse_groups(raw, n)
            if idx_groups is not None:
                break
        if attempt < max_retries:
            _time.sleep(2 * attempt)

    gt_groups  = item["ground_truth"].get(f"{dim}_groups", [])
    eval_valid = dim in item["eval_dims"]

    result: dict = {
        "item_id":               item["item_id"],
        "source_survey":         item["source_survey"],
        "n_papers":              n,
        "dim":                   dim,
        "model":                 model,
        "content_mode":          mode,
        "sample_idx":            sample_idx,
        "predicted_groups":      idx_groups,
        "predicted_groups_rich": rich_groups,
        "ground_truth_groups":   gt_groups,
        "eval_valid":            eval_valid,
        "parse_error":           idx_groups is None and not error,
        "llm_error":             error,
        "raw_response":          raw,
    }
    if reasoning:
        result["reasoning_content"] = reasoning

    if eval_valid and idx_groups is not None:
        result["metrics"] = eval_groups(idx_groups, gt_groups)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",         default="data/grouping.json")
    parser.add_argument("--models",        nargs="+", required=True)
    parser.add_argument("--content-modes", nargs="+", default=["abstract"],
                        choices=["abstract", "abstract+intro", "full"])
    parser.add_argument("--dims",          nargs="+", default=["task", "problem", "method"],
                        choices=["task", "problem", "method"])
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--base-url",      default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key",       default=DEFAULT_API_KEY)
    parser.add_argument("--workers",       type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout",       type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--temperature",   type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens",    type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-tokens-per-paper-abstract", type=int,
                        default=DEFAULT_MAX_ABSTRACT_TOKENS)
    parser.add_argument("--max-tokens-per-paper-intro",    type=int,
                        default=DEFAULT_MAX_INTRO_TOKENS)
    parser.add_argument("--max-tokens-per-paper-full",     type=int,
                        default=DEFAULT_MAX_FULL_TOKENS)
    parser.add_argument("--n-samples",      type=int, default=1,
                        help="Number of independent LLM samples per (item, dim) task "
                             "(default 1). When >1, mean±std is reported across samples.")
    parser.add_argument("--thinking-budget", type=int, default=0,
                        help="Enable thinking with this budget_tokens (0=disabled)")
    parser.add_argument("--limit",         type=int, default=None)
    args = parser.parse_args()

    mode_max_tokens = {
        "abstract":       args.max_tokens_per_paper_abstract,
        "abstract+intro": args.max_tokens_per_paper_intro,
        "full":           args.max_tokens_per_paper_full,
    }

    with open(args.input) as f:
        items: list[dict] = json.load(f)
    if args.limit:
        items = items[: args.limit]

    templates = {dim: Path(PROMPT_FILES[dim]).read_text() for dim in args.dims}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    SAVE_EVERY = 10  # incremental save interval

    for model in args.models:
        model_slug = model.replace("/", "_")
        for mode in args.content_modes:
            out_path = out_dir / f"{model_slug}_{mode}.json"

            # ── Item-level resume: load existing results and build done_keys ──
            existing_results: list[dict] = []
            done_keys: set[tuple[str, str, int]] = set()
            if out_path.exists():
                try:
                    with open(out_path, encoding="utf-8") as f:
                        existing_data = json.load(f)
                    all_existing = existing_data.get("results", [])
                    # Skip successful results only; retry both llm_error and parse_error entries.
                    existing_results = [
                        r for r in all_existing
                        if not r.get("llm_error") and not r.get("parse_error")
                    ]
                    done_keys = {
                        (r["item_id"], r["dim"], r.get("sample_idx", 0))
                        for r in existing_results
                        if "item_id" in r and "dim" in r
                    }
                except Exception as e:
                    print(f"  [warn] Could not load existing results from {out_path}: {e}")
                    existing_results = []
                    done_keys = set()

            # Build task list: only dims in item["eval_dims"] (others have no GT)
            # For n_samples > 1, each (item, dim) is repeated n_samples times.
            all_tasks = [
                (item, dim, s_idx)
                for s_idx in range(args.n_samples)
                for item in items
                for dim in args.dims
                if dim in item.get("eval_dims", [])
            ]
            tasks = [
                (item, dim, s_idx)
                for item, dim, s_idx in all_tasks
                if (item["item_id"], dim, s_idx) not in done_keys
            ]

            n_existing = len(existing_results)
            print(f"\nModel={model}  mode={mode}  items={len(items)}  "
                  f"n_samples={args.n_samples}  tasks_total={len(all_tasks)}")
            print(f"  Loaded {n_existing} existing results, {len(tasks)} remaining")

            if not tasks:
                # Nothing to do — recompute summary and exit
                summary = _aggregate(existing_results, args.dims, args.n_samples)
                out_data = {"model": model, "content_mode": mode,
                            "n_samples": args.n_samples,
                            "summary": summary, "results": existing_results}
                with open(out_path, "w") as f:
                    json.dump(out_data, f, indent=2, ensure_ascii=False)
                _print_summary(summary, model, mode)
                print(f"  → {out_path} (no new tasks)")
                continue

            results: list[dict] = list(existing_results)

            def _worker(args_tuple):
                item, dim, s_idx = args_tuple
                return process_one(
                    item, dim, templates[dim], mode,
                    client, model,
                    args.timeout, args.temperature, args.max_tokens,
                    mode_max_tokens[mode],
                    sample_idx=s_idx,
                    thinking_budget=args.thinking_budget,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_worker, t): t for t in tasks}
                done = 0
                for fut in concurrent.futures.as_completed(futures):
                    done += 1
                    res = fut.result()
                    results.append(res)
                    if done % SAVE_EVERY == 0 or done == len(tasks):
                        print(f"  [{done}/{len(tasks)}]", flush=True)
                        # Incremental save
                        summary = _aggregate(results, args.dims, args.n_samples)
                        out_data = {"model": model, "content_mode": mode,
                                    "n_samples": args.n_samples,
                                    "summary": summary, "results": results}
                        with open(out_path, "w") as f:
                            json.dump(out_data, f, indent=2, ensure_ascii=False)

            # Final aggregate + save
            summary = _aggregate(results, args.dims, args.n_samples)
            out_data = {"model": model, "content_mode": mode,
                        "n_samples": args.n_samples,
                        "summary": summary, "results": results}
            with open(out_path, "w") as f:
                json.dump(out_data, f, indent=2, ensure_ascii=False)

            _print_summary(summary, model, mode)
            print(f"  → {out_path}")


def _aggregate(results: list[dict], dims: list[str], n_samples: int = 1) -> dict:
    import statistics as _stats

    summary: dict = {}
    for dim in dims:
        dim_results = [r for r in results if r["dim"] == dim and r["eval_valid"]]
        if not dim_results:
            summary[dim] = {"n_items": 0, "n_eval": 0}
            continue

        if n_samples > 1:
            # Group by sample_idx; compute macro-avg F1/P/R per sample
            sample_map: dict[int, dict[str, list[float]]] = {}
            for r in dim_results:
                s_idx = r.get("sample_idx", 0)
                if r.get("metrics"):
                    if s_idx not in sample_map:
                        sample_map[s_idx] = {"f1": [], "p": [], "r": []}
                    sample_map[s_idx]["f1"].append(r["metrics"]["f1"])
                    sample_map[s_idx]["p"].append(r["metrics"]["precision"])
                    sample_map[s_idx]["r"].append(r["metrics"]["recall"])

            sample_f1 = [sum(v["f1"]) / len(v["f1"]) for _, v in sorted(sample_map.items())]
            sample_p  = [sum(v["p"])  / len(v["p"])  for _, v in sorted(sample_map.items())]
            sample_r  = [sum(v["r"])  / len(v["r"])  for _, v in sorted(sample_map.items())]

            n_items = len({r["item_id"] for r in dim_results}) // n_samples \
                      if n_samples else len({r["item_id"] for r in dim_results})
            summary[dim] = {
                "n_items":      n_items,
                "n_samples":    len(sample_map),
                "sample_f1s":   [round(f, 4) for f in sample_f1],
                "avg_f1_mean":  round(_stats.mean(sample_f1), 4) if sample_f1 else 0.0,
                "avg_f1_std":   round(_stats.stdev(sample_f1), 4) if len(sample_f1) > 1 else 0.0,
                "avg_p_mean":   round(_stats.mean(sample_p),  4) if sample_p  else 0.0,
                "avg_r_mean":   round(_stats.mean(sample_r),  4) if sample_r  else 0.0,
            }
        else:
            evaled = [r for r in dim_results if r.get("metrics")]
            if not evaled:
                summary[dim] = {"n_items": 0, "n_eval": 0}
                continue
            p  = sum(r["metrics"]["precision"] for r in evaled) / len(evaled)
            r_ = sum(r["metrics"]["recall"]    for r in evaled) / len(evaled)
            f1 = sum(r["metrics"]["f1"]        for r in evaled) / len(evaled)
            summary[dim] = {
                "n_items":       len(dim_results),
                "n_eval":        len(evaled),
                "parse_errors":  sum(1 for r in dim_results if r.get("parse_error")),
                "avg_precision": round(p, 4),
                "avg_recall":    round(r_, 4),
                "avg_f1":        round(f1, 4),
            }
    return summary


def _print_summary(summary: dict, model: str, mode: str):
    print(f"\n  === {model} | {mode} ===")
    for dim, s in summary.items():
        if s.get("n_samples", 1) > 1:
            f1_mean = s.get("avg_f1_mean", 0.0)
            f1_std  = s.get("avg_f1_std",  0.0)
            each    = [round(f, 3) for f in s.get("sample_f1s", [])]
            print(f"  {dim:8s}: F1={f1_mean:.3f}±{f1_std:.3f}  "
                  f"P={s.get('avg_p_mean', 0):.3f}  R={s.get('avg_r_mean', 0):.3f}  "
                  f"(n_items={s['n_items']}, n_samples={s['n_samples']}, "
                  f"per-sample={each})")
        elif s.get("n_eval", 0) == 0:
            print(f"  {dim:8s}: no eval items")
        else:
            print(f"  {dim:8s}: P={s['avg_precision']:.3f}  "
                  f"R={s['avg_recall']:.3f}  F1={s['avg_f1']:.3f}"
                  f"  (n={s['n_eval']}, parse_err={s.get('parse_errors', 0)})")


if __name__ == "__main__":
    main()
