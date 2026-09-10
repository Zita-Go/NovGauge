"""Run dimension-specific pairwise similarity inference. Use --help for options."""

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

BASE        = Path(__file__).parent
REPO        = BASE.parent.parent
PROMPT_DIR  = BASE.parent / "prompts"
CACHE_PATH  = REPO / "bench/output/cache/paper_content_cache_human.json"

PROMPT_FILES = {
    "task":    PROMPT_DIR / "similarity_task_prompt.txt",
    "problem": PROMPT_DIR / "similarity_problem_prompt.txt",
    "method":  PROMPT_DIR / "similarity_method_prompt.txt",
}

DEFAULT_BASE_URL    = os.getenv("NOVGAUGE_BASE_URL")
DEFAULT_API_KEY     = os.getenv("NOVGAUGE_API_KEY")
DEFAULT_WORKERS     = 8
DEFAULT_TIMEOUT     = 120
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS  = 4096
DEFAULT_THINKING_BUDGET = 8000

# Per-paper token limits (cl100k_base)
DEFAULT_MAX_ABSTRACT_TOKENS = 1_000
DEFAULT_MAX_INTRO_TOKENS    = 4_000
DEFAULT_MAX_FULL_TOKENS     = 16_000

# Tokenizer (lazy-loaded cl100k_base; character fallback if unavailable)
_tokenizer: Any = None  # tiktoken encoding or False


def _get_tokenizer() -> Any | None:
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken  # type: ignore[import-not-found]
            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tokenizer = False
    return None if _tokenizer is False else _tokenizer


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens.
    If tiktoken is unavailable, approximate the limit using 4 characters per token."""
    enc = _get_tokenizer()
    if enc is None:
        return text[: max_tokens * 4]
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    return enc.decode(ids[:max_tokens])

# Markdown heading parsing
def _normalize_heading_text(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"^\d+(?:\.\d+)*[\.\)]?\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :.-")


def _effective_heading_level(markdown_level: int, heading_text: str) -> int:
    stripped = heading_text.strip()
    m = re.match(r"^(\d+(?:\.\d+)*)[\.\)]?(?:\s|$)", stripped)
    if m:
        return m.group(1).count(".") + 1
    m = re.match(r"^([A-Z])[\.\)]?(?:\s|$)", stripped)
    if m:
        return 1
    return markdown_level


def _iter_markdown_headings(lines: list[str]) -> list[dict[str, Any]]:
    pseudo = (
        r"^(algorithm|figure|fig\.|table|theorem|lemma|corollary|proposition|remark|example)\b",
    )
    headings: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        m = re.match(r"^(#+)\s*(.*)$", line)
        if not m:
            continue
        heading_text = m.group(2).strip()
        normalized = _normalize_heading_text(heading_text)
        if any(re.match(p, normalized, flags=re.IGNORECASE) for p in pseudo):
            continue
        headings.append({
            "idx": idx,
            "level": _effective_heading_level(len(m.group(1)), heading_text),
            "text": heading_text,
            "normalized": normalized,
        })
    return headings


def _extract_heading_block(lines: list[str], headings: list[dict], pos: int) -> tuple[int, int]:
    heading = headings[pos]
    start = int(heading["idx"])
    level = int(heading["level"])
    end = len(lines)
    for nxt in headings[pos + 1:]:
        if int(nxt["level"]) <= level:
            end = int(nxt["idx"])
            break
    return start, end


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _join_ranges(lines: list[str], ranges: list[tuple[int, int]]) -> str:
    chunks = ["\n".join(lines[s:e]).strip() for s, e in ranges]
    return "\n\n".join(c for c in chunks if c).strip()


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


# Content extraction
_INTRO_PATTERNS = (r"^introduction\b", r"^1\s+introduction\b")

_REF_PATTERNS = (r"^references?\b", r"^bibliography\b")


def _normalize_ocr(text: str) -> str:
    """Collapse OCR-spaced letters such as 'R E F E R E N C E S' into a word."""
    # Remove single spaces between individual letters (such as 'R E F ...') before matching.
    collapsed = re.sub(r"(?<=\S) (?=\S)", "", text)
    return collapsed.lower().strip()


def _find_ref_line(lines: list[str], headings: list[dict]) -> int | None:
    for h in headings:
        norm = str(h["normalized"])
        # Try a direct match first, then match after collapsing OCR-spaced letters.
        if any(re.match(p, norm, flags=re.IGNORECASE) for p in _REF_PATTERNS):
            return int(h["idx"])
        ocr_norm = _normalize_ocr(str(h["text"]))
        if any(re.match(p, ocr_norm, flags=re.IGNORECASE) for p in _REF_PATTERNS):
            return int(h["idx"])
    return None


def _find_intro_idx(headings: list[dict]) -> int | None:
    for idx, h in enumerate(headings):
        if int(h["level"]) != 1:
            continue
        norm = str(h["normalized"])
        if any(re.match(p, norm, flags=re.IGNORECASE) for p in _INTRO_PATTERNS):
            return idx
        # Handle OCR-spaced headings like "# 1 IN T RO D U C T I O N"
        ocr = _normalize_ocr(str(h["text"]))
        if re.match(r"^[\d.]*introduction$", ocr, re.IGNORECASE):
            return idx
    return None


def strip_references(text: str) -> str:
    """Remove the References section and all subsequent content."""
    lines = text.splitlines()
    headings = _iter_markdown_headings(lines)
    ref_line = _find_ref_line(lines, headings)
    if ref_line is not None:
        return "\n".join(lines[:ref_line]).strip()
    return text


def extract_intro_section(md_text: str) -> str:
    """Extract the Introduction body from Markdown; return an empty string if absent."""
    lines = md_text.splitlines()
    headings = _iter_markdown_headings(lines)
    intro_idx = _find_intro_idx(headings)
    if intro_idx is None:
        return ""
    s, e = _extract_heading_block(lines, headings, intro_idx)
    return "\n".join(lines[s:e]).strip()


def get_content(paper: dict, mode: str, max_tokens: int) -> tuple[str, str]:
    """Return (content_label, text), using cached content where available.
    Limit each paper to max_tokens."""
    abstract = (paper.get("abstract") or "").strip()
    md_path  = paper.get("markdown_path")

    if mode == "abstract":
        return "abstract", truncate_by_tokens(abstract, max_tokens)

    # abstract+intro: use pre-extracted cache; full: read markdown directly
    md_text = ""
    if md_path:
        full_path = REPO / md_path
        if full_path.exists():
            md_text = full_path.read_text(encoding="utf-8", errors="ignore")

    if mode == "abstract+intro":
        if md_path:
            cache = _load_cache()
            entry = cache.get(md_path)
            if entry:
                text = entry.get("abstract+intro") or abstract
                return "abstract+introduction", truncate_by_tokens(text, max_tokens)
        # Fallback: runtime extraction
        intro = extract_intro_section(md_text) if md_text else ""
        combined = f"Abstract: {abstract}\n\n{intro}" if intro else abstract
        return "abstract+introduction", truncate_by_tokens(combined, max_tokens)

    else:  # full
        text = strip_references(md_text) if md_text else abstract
        return "full text", truncate_by_tokens(text, max_tokens)


# Prompt construction
def build_prompt(template: str, paper_a: dict, paper_b: dict,
                 mode: str, max_tokens_per_paper: int) -> str:
    ct_a, c_a = get_content(paper_a, mode, max_tokens_per_paper)
    ct_b, c_b = get_content(paper_b, mode, max_tokens_per_paper)
    return (template
            .replace("<TITLE_A>", paper_a.get("title") or "")
            .replace("<CONTENT_TYPE_A>", ct_a)
            .replace("<CONTENT_A>", c_a)
            .replace("<TITLE_B>", paper_b.get("title") or "")
            .replace("<CONTENT_TYPE_B>", ct_b)
            .replace("<CONTENT_B>", c_b))


# LLM calls
def _fix_json_quotes(s: str) -> str:
    """Replace unescaped inner double-quotes in JSON string values with single quotes."""
    def _fix(m: re.Match) -> str:
        inner = re.sub(r'(?<!\\)"', "'", m.group(1))
        return f'"{inner}"'
    return re.sub(r'"((?:[^"\\]|\\.)*)"', _fix, s)


def _try_parse(s: str, key: str) -> dict | None:
    """Try strict then quote-fixed JSON parse; return object if key present."""
    try:
        # PDF/OCR text can contain literal control characters (for example
        # \x02 from malformed font mappings).  They are valid evidence text
        # but invalid under the JSON decoder's default strict mode.
        obj = json.loads(s, strict=False)
        return obj if key in obj else None
    except json.JSONDecodeError:
        pass
    try:
        obj = json.loads(_fix_json_quotes(s), strict=False)
        return obj if key in obj else None
    except json.JSONDecodeError:
        return None


def parse_response(text: str) -> dict | None:
    # Try all code blocks, take the last valid one (model may revise its answer)
    last_obj: dict | None = None
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        obj = _try_parse(m.group(1), "is_similar")
        if obj is not None:
            last_obj = obj
    if last_obj is not None:
        return last_obj

    # Fallback: string-aware scan for JSON objects containing "is_similar".
    # Skips braces inside JSON string literals to avoid miscounting when
    # evidence text contains curly braces (e.g. math formulas).
    for m in re.finditer(r'\{', text):
        start = m.start()
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    obj = _try_parse(text[start: i + 1], "is_similar")
                    if obj is not None:
                        last_obj = obj
                    break
    if last_obj is not None:
        return last_obj

    # Final fallback: try json.loads on the full stripped text
    obj = _try_parse(text.strip(), "is_similar")
    return obj


import re as _re

# Match local vLLM Qwen3 names (qwen3-8b / qwen3-32b / qwen3.5-27b).
_LOCAL_QWEN3_RE = _re.compile(r"^qwen3(\.\d+)?-\d+b$", _re.IGNORECASE)

# Models requiring streaming to capture reasoning_content when the proxy filters it from non-streamed responses.
_STREAMING_MODELS: set[str] = {"anthropic/claude-opus-4.7"}
# Map configured model identifiers to the names used in API requests.
_MODEL_API_NAME_MAP: dict[str, str] = {"anthropic/claude-opus-4.7": "claude-opus-4-7"}


def _thinking_extra_body(model: str, thinking_budget: int) -> dict | None:
    """Build model-specific reasoning parameters from the thinking switch."""
    m = model.lower()

    # :thinking variants encode reasoning settings in the model name (for example, claude-sonnet-4.6:thinking); omit extra_body.
    if m.endswith("-thinking") or m.endswith(":thinking"):
        return None

    # DeepSeek enables reasoning by default; explicitly disable it when budget=0.
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

    # Qwen API uses the maximum reasoning length when thinking_budget is omitted.
    if "qwen" in m:
        return {"enable_thinking": True}

    if "claude-opus-4.7" in m or "claude-opus-4-7" in m:
        # For streaming, display=summarized overrides omitted to return reasoning summaries.
        return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "max"}}

    if "claude" in m:
        return {"thinking": {"type": "adaptive"}}

    return {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}


def _extract_reasoning(msg) -> str:
    """Extract reasoning_content from the response, also supporting <think> tags."""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    if not reasoning:
        m = re.search(r"<think>(.*?)</think>", msg.content or "", re.DOTALL)
        if m:
            reasoning = m.group(1).strip()
    return reasoning


def _call_llm_streaming(
    client: OpenAI, api_model: str, prompt: str,
    timeout: int, max_tokens: int, extra_body: dict | None,
) -> tuple[str, str | None, str]:
    """Stream the response and concatenate content and reasoning_content deltas.
    Set temperature=1 for Claude reasoning mode.
    Return (raw_text, error_str, reasoning_content).
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


def call_llm(client: OpenAI, model: str, prompt: str,
             timeout: int, temperature: float, max_tokens: int,
             max_retries: int = 3,
             thinking_budget: int = DEFAULT_THINKING_BUDGET) -> tuple[dict | None, str | None, str | None]:
    """Return (parsed_dict, error_str, reasoning_content)."""
    api_model  = _MODEL_API_NAME_MAP.get(model, model)
    use_stream = model in _STREAMING_MODELS
    extra      = _thinking_extra_body(model, thinking_budget)

    for attempt in range(1, max_retries + 1):
        try:
            if use_stream:
                text, err, reasoning = _call_llm_streaming(
                    client, api_model, prompt, timeout, max_tokens, extra,
                )
                if err:
                    raise Exception(err)
            else:
                kwargs: dict = dict(
                    model=api_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=timeout,
                )
                if max_tokens > 0:
                    kwargs["max_tokens"] = max_tokens
                if extra:
                    kwargs["extra_body"] = extra
                resp = client.chat.completions.create(**kwargs)
                msg  = resp.choices[0].message
                text = msg.content or ""
                reasoning = _extract_reasoning(msg)

            # Remove <think> blocks from content before parsing JSON.
            text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            parsed = parse_response(text_clean)
            if parsed is not None:
                return parsed, None, reasoning
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                return None, f"JSON parsing failed: {text_clean[:300]}", reasoning
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                return None, str(e), None
    return None, "Maximum retry attempts exceeded", None


# Individual task
def run_one(
    item: dict, dim: str, ground_truth: int, template: str,
    client: OpenAI, model: str, mode: str,
    timeout: int, temperature: float, max_tokens: int,
    max_tokens_per_paper: int,
    n_samples: int = 1,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> dict:
    paper_a = item.get("submitted_paper") or item.get("paper_a") or {}
    paper_b = item.get("prior_work") or item.get("negative_paper") or item.get("paper_b") or {}
    prompt  = build_prompt(template, paper_a, paper_b, mode, max_tokens_per_paper)

    samples, errors = [], []
    for _ in range(n_samples):
        result, err, reasoning = call_llm(
            client, model, prompt, timeout, temperature, max_tokens,
            thinking_budget=thinking_budget,
        )
        if result is not None:
            p = 1 if result.get("is_similar") else 0
            sample: dict = {"pred": p, "llm_output": result}
            if reasoning:
                sample["reasoning_content"] = reasoning
            samples.append(sample)
        else:
            errors.append(err)

    if not samples:
        # All attempts failed
        return {
            "pair_id":      item.get("pair_id", ""),
            "title_a":      paper_a.get("title", ""),
            "title_b":      paper_b.get("title", ""),
            "dim":          dim,
            "label":        ground_truth,
            "pred":         None,
            "correct":      None,
            "pass_at_1":    None,
            "samples":      [],
            "error":        "; ".join(errors),
            "model":        model,
            "content_mode": mode,
        }

    preds = [s["pred"] for s in samples]
    # Use sample 0 as the primary prediction for Stage 2/3 evidence and reasoning checks.
    pred = preds[0]

    return {
        "pair_id":      item.get("pair_id", ""),
        "title_a":      paper_a.get("title", ""),
        "title_b":      paper_b.get("title", ""),
        "dim":          dim,
        "label":        ground_truth,
        "pred":         pred,
        "correct":      (pred == ground_truth),
        "samples":      samples,
        "error":        ("; ".join(errors)) if errors else None,
        "model":        model,
        "content_mode": mode,
    }


# Statistics
def _dim_f1_from_preds(dim_recs: list[dict], s_idx: int) -> tuple[float, float, float, float]:
    """Compute (acc, prec, rec, f1) using samples[s_idx].pred for each record."""
    tp = fp = fn = tn = 0
    for r in dim_recs:
        samps = r.get("samples", [])
        if s_idx >= len(samps):
            continue
        p  = samps[s_idx]["pred"]
        lb = r.get("label")
        if lb is None:
            continue
        if   lb == 1 and p == 1: tp += 1
        elif lb == 1 and p == 0: fn += 1
        elif lb == 0 and p == 1: fp += 1
        else:                    tn += 1
    n    = tp + fp + fn + tn
    acc  = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return acc, prec, rec, f1


def print_summary(results: list[dict], model: str, mode: str) -> None:
    import math
    import statistics as _stats
    from collections import defaultdict

    n_samples_max = max((len(r.get("samples", [])) for r in results), default=1)
    is_multi = n_samples_max > 1

    if is_multi:
        print(f"\n  [{mode}] ===== {model}  (n_samples={n_samples_max}, mean±std) =====")
        print(f"  {'dim':8s}  {'F1 mean':>8}  {'±std':>6}  {'Acc mean':>9}  per-sample F1")
        print(f"  {'-'*65}")
        for dim in ["method", "problem", "task"]:
            dim_recs = [r for r in results if r["dim"] == dim and r.get("samples")]
            if not dim_recs:
                continue
            sample_f1s, sample_accs = [], []
            for s_idx in range(n_samples_max):
                acc, prec, rec, f1 = _dim_f1_from_preds(dim_recs, s_idx)
                if not math.isnan(f1):
                    sample_f1s.append(f1)
                if not math.isnan(acc):
                    sample_accs.append(acc)
            if not sample_f1s:
                continue
            mean_f1 = _stats.mean(sample_f1s)
            std_f1  = _stats.stdev(sample_f1s) if len(sample_f1s) > 1 else 0.0
            mean_acc = _stats.mean(sample_accs) if sample_accs else float("nan")
            each = [round(f, 3) for f in sample_f1s]
            print(f"  {dim:8s}  {mean_f1:8.3f}  ±{std_f1:.3f}  {mean_acc:9.3f}  {each}")
    else:
        dim_stats: dict[str, dict] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "err": 0}
        )
        for r in results:
            d = r["dim"]
            if r["pred"] is None:
                dim_stats[d]["err"] += 1
                continue
            lb, pr = r["label"], r["pred"]
            if   lb == 1 and pr == 1: dim_stats[d]["tp"] += 1
            elif lb == 1 and pr == 0: dim_stats[d]["fn"] += 1
            elif lb == 0 and pr == 1: dim_stats[d]["fp"] += 1
            else:                     dim_stats[d]["tn"] += 1

        print(f"\n  [{mode}] ===== {model} =====")
        print(f"  {'dim':8s}  {'Acc':>6}  {'Prec':>6}  {'Rec(↑)':>8}  {'F1':>6}  "
              f"{'TP':>4}  {'FN':>4}  {'FP':>4}  {'ERR':>4}")
        print(f"  {'-'*72}")
        for dim in ["method", "problem", "task"]:
            s = dim_stats.get(dim)
            if not s:
                continue
            n = s["tp"] + s["fp"] + s["tn"] + s["fn"]
            if n == 0:
                continue
            acc  = (s["tp"] + s["tn"]) / n
            prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
            rec  = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            print(f"  {dim:8s}  {acc:6.3f}  {prec:6.3f}  {rec:8.3f}  {f1:6.3f}  "
                  f"{s['tp']:4d}  {s['fn']:4d}  {s['fp']:4d}  {s['err']:4d}")


# Inference for one (model × mode) combination
def run_single(
    tasks: list[tuple[dict, str, int]],
    templates: dict[str, str],
    model: str,
    mode: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{mode.replace('+', '_')}.json"

    # Resume completed entries; retry entries without enough successful samples.
    done_ids: set[tuple] = set()
    results:  list[dict] = []
    if out_file.exists():
        try:
            all_results = json.load(open(out_file, encoding="utf-8"))
            done_ids = {(r["pair_id"], r["dim"]) for r in all_results
                        if len(r.get("samples") or []) >= args.n_samples}
            results   = [r for r in all_results
                         if len(r.get("samples") or []) >= args.n_samples]
            print(f"  [{mode}] Resuming with {len(results)} successful results")
        except Exception:
            pass

    pending = [
        (item, dim, gt) for item, dim, gt in tasks
        if (item.get("pair_id", ""), dim) not in done_ids
    ]
    print(f"  [{ mode}] Pending inference tasks: {len(pending)}")

    if not pending:
        print(f"  [{mode}] All tasks are complete; skipping")
        print_summary(results, model, mode)
        return

    client    = OpenAI(api_key=args.api_key, base_url=args.base_url)
    completed = 0

    mode_token_limits = {
        "abstract":       args.max_abstract_tokens,
        "abstract+intro": args.max_intro_tokens,
        "full":           args.max_full_tokens,
    }
    max_tokens_per_paper = mode_token_limits[mode]
    print(f"  [{mode}] Token limit per paper: {max_tokens_per_paper:,}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                run_one, item, dim, gt, templates[dim], client, model,
                mode, args.timeout, args.temperature, args.max_tokens,
                max_tokens_per_paper, args.n_samples, args.thinking_budget,
            ): (item, dim, gt)
            for item, dim, gt in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            rec = future.result()
            results.append(rec)
            completed += 1
            total_done = len(done_ids) + completed
            status = "✓" if rec["correct"] else ("✗" if rec["correct"] is False else "?")
            print(f"    [{total_done:4d}/{len(tasks)}] {status} "
                  f"dim={rec['dim']} label={rec['label']} pred={rec['pred']} | "
                  f"{rec['pair_id'][:55]}")
            if completed % 20 == 0 or completed == len(pending):
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [{mode}] Results saved to: {out_file}")
    print_summary(results, model, mode)


# Main workflow
def run(args: argparse.Namespace) -> None:
    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)

    templates = {
        dim: PROMPT_FILES[dim].read_text(encoding="utf-8")
        for dim in args.dims
    }

    # Build the task list.
    tasks: list[tuple[dict, str, int]] = []
    for item in items:
        sim_dims = set(item.get("similar_dimensions", []))
        lbs      = item.get("labels", {})
        for dim in args.dims:
            if sim_dims:
                gt = 1 if dim in sim_dims else None
            elif lbs:
                gt = lbs.get(dim)
            else:
                gt = None
            if gt is None:
                continue
            if args.neg_only and gt != 0:
                continue   # Skip label=1 dimensions; keep only confirmed negatives.
            tasks.append((item, dim, gt))

    if args.limit:
        tasks = tasks[:args.limit]

    print(f"Input file: {args.input}")
    print(f"Dimensions:     {args.dims}")
    print(f"Content modes:     {args.content_modes}")
    print(f"Total tasks:   {len(tasks)} per content mode\n")

    for model in args.models:
        model_dir = Path(args.output_dir) / model.replace("/", "_")
        print(f"{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        for mode in args.content_modes:
            run_single(tasks, templates, model, mode, model_dir, args)
            print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input",          default="data/positives.json")
    p.add_argument("--models",         nargs="+", default=["openai/gpt-5.4"])
    p.add_argument("--dims",           nargs="+",
                   choices=["task", "problem", "method"],
                   default=["task", "problem", "method"])
    p.add_argument("--content-modes",  nargs="+",
                   choices=["abstract", "abstract+intro", "full"],
                   default=["full"])
    p.add_argument("--output-dir",     default="results")
    p.add_argument("--base-url",       default=DEFAULT_BASE_URL)
    p.add_argument("--api-key",        default=DEFAULT_API_KEY)
    p.add_argument("--workers",        type=int,   default=DEFAULT_WORKERS)
    p.add_argument("--timeout",        type=int,   default=DEFAULT_TIMEOUT)
    p.add_argument("--temperature",    type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens",          type=int, default=DEFAULT_MAX_TOKENS,
                   help="Maximum number of LLM output tokens")
    p.add_argument("--max-abstract-tokens", type=int, default=DEFAULT_MAX_ABSTRACT_TOKENS,
                   help="Per-paper content token limit in abstract mode")
    p.add_argument("--max-intro-tokens",    type=int, default=DEFAULT_MAX_INTRO_TOKENS,
                   help="Per-paper content token limit in abstract+intro mode")
    p.add_argument("--max-full-tokens",     type=int, default=DEFAULT_MAX_FULL_TOKENS,
                   help="Per-paper content token limit in full mode")
    p.add_argument("--limit",               type=int, default=None)
    p.add_argument("--neg-only",            action="store_true",
                   help="Keep only label=0 tasks for negative-example evaluation")
    p.add_argument("--n-samples",           type=int, default=1,
                   help="Independent samples per task; with >1, compute F1 per sample and report mean +/- std")
    p.add_argument("--thinking-budget",     type=int, default=DEFAULT_THINKING_BUDGET,
                   help="Enable model-specific reasoning (default: 8000; 0 omits or disables it where supported)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
