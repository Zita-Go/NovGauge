# Evaluation protocol and outputs

The framework consumes existing benchmark labels and user-provided Markdown.
Its four stages are:

1. **Similarity metrics.** Pairwise classification reports accuracy, precision,
   recall and F1 per dimension. Grouping converts predicted and reference groups
   to paper pairs and reports macro and micro metrics.
2. **Evidence checking.** Evidence is checked against paper text and title. The
   implementation distinguishes exact matches, extraction artifacts, soft matches
   using character 4-gram recall, and hallucinations. This soft match is not
   ROUGE-L. The existing soft-match threshold is 0.75.
3. **Reasoning checking.** An LLM judge checks whether the supplied evidence
   supports the model's reason. Failed requests or malformed responses may leave
   a judgment unscored; inspect the detailed result files as well as the summary.
4. **Verified F1.** A true-positive prediction receives verification credit only
   when its evidence and reasoning pass the corresponding checks. The other
   predicted positives still contribute to the precision denominator.

For multiple samples, the evaluators compute per-sample metrics and report their
mean and sample standard deviation where supported. This is not a majority vote.
The JSON metric schema differs between single- and multiple-sample runs; for
example pairwise Stage 1 uses `f1` or `f1_mean` / `f1_per_sample` respectively.

## Entry-point presets

The numerical settings below preserve the original implementation. Use explicit
settings when comparing models or switching entry points.

| Setting | Pairwise Python | Pairwise shell | Grouping Python | Grouping shell |
| --- | --- | --- | --- | --- |
| Default content modes | All three | `full` | `abstract` | `full` |
| Temperature | 0.6 | 0.6 | 0.2 | 0.6 |
| Maximum output tokens | 4096 | 4096 | 1024 | 8192 |
| Abstract tokens per paper | 1000 | 1000 | 800 | 1000 |
| Abstract + intro tokens per paper | 4000 | 4000 | 3000 | 4000 |
| Full-text tokens per paper | 16000 | 16000 | 12000 | 12000 |
| Samples | 1 | 1 | 1 | 1 |
| Thinking budget flag | 0 | 8000 | 0 | 8000 |

Integrated Python evaluators default to 512 judge output tokens. The shell
workflows pass 4096. The existing model-specific thinking logic maps the thinking
flag to provider-specific parameters; its numeric value is not a universal
reasoning-token limit. Model aliases and streaming behavior also depend on the
endpoint. The framework retains those request rules for comparison with earlier
runs.

## Output files

`MODEL_SLUG` is the model identifier with `/` replaced by `_`.

| Task | Inference output below the supplied output directory |
| --- | --- |
| Pairwise, abstract | `MODEL_SLUG/abstract.json` |
| Pairwise, abstract + intro | `MODEL_SLUG/abstract_intro.json` |
| Pairwise, full | `MODEL_SLUG/full.json` |
| Grouping | `MODEL_SLUG_MODE.json`, retaining `+` in `abstract+intro` |

The shell workflows write their evaluation summary and `metrics.json` under
`RUN_DIR/eval_pipeline/`, alongside evidence and reasoning-check details.
Both `.env` files and generated results are ignored by Git. Results may contain
paper excerpts; review them separately before distributing them.

## Reproducibility and preserved behavior

- Supply readable Markdown and abstracts for the selected mode. A missing or
  empty Markdown input may fall back to the abstract in the core runners; the
  shell checks dataset files, not the contents of each paper record.
- Pairwise and grouping use separate introduction extraction rules. They also
  prefer a local abstract/intro cache when present. Different Markdown or caches
  can change prompts and scores.
- Evidence checking can read more text than the token-truncated inference prompt.
  The current verification scores therefore do not guarantee that every accepted
  excerpt was inside the exact truncated text shown to the model.
- Missing and failed samples affect the evaluation denominator. Pairwise and
  grouping handling differs; compare request failures, parse errors and sample
  coverage alongside the reported scores. A completed inference command does not
  guarantee that every request succeeded; inspect the per-record errors.
- Resume logic checks record identities and sample counts, not a complete hash
  of data, prompts and configuration. Use a new output directory when those
  inputs change. The framework does not yet save a complete run manifest.
- Run Stages 2, 3 and 4 together for verified F1. Selecting Stage 4 alone does not
  trigger it, and skipped or stale earlier stages can make its output incomplete.
- JSON Boolean fields should be actual `true` / `false` values. The existing
  parsers use Python truth-value conversion in some paths; strings such as
  `"false"` are not a supported substitute.

These behaviors are documented without changing the historical scoring rules.
Reproducing the published scores requires the historical text and configuration
as well as access to the corresponding model endpoints.
