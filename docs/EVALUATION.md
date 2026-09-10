# Evaluation

## Stages

1. **Similarity:** accuracy, precision, recall, and F1; grouping is scored using paper pairs.
2. **Evidence:** check quoted evidence against the source text.
3. **Reasoning:** an LLM judge checks whether the evidence supports the reason.
4. **Verified F1:** credit true positives only when evidence and reasoning pass.

Multiple samples are scored separately and summarized as mean and standard deviation.

## Settings

Both tasks and both entry points default to `full`, temperature **0.6**,
**4096** output tokens, **1** sample, and **thinking enabled**.

| Content mode | Input tokens per paper |
| --- | ---: |
| `abstract` | 1000 |
| `abstract+intro` | 4000 |
| `full` | 16000 |

`--thinking-budget` defaults to 8000 and selects model-specific reasoning settings,
not a universal token budget. Provider adapters may override settings; the Claude
streaming path uses temperature 1. Judge settings are separate: temperature 0,
512 output tokens in Python or 4096 in shell, with no explicit thinking setting.

For shell runs, override settings through variables such as `CONTENT_MODE`,
`EVAL_N_SAMPLES`, `EVAL_MAX_TOKENS`, and `WORKERS`. Use each Python entry point's
`--help` for its flags. Setup and example commands are in the [README](../README.md).

## Saved results

The shell workflows save predictions under `RUN_DIR` and evaluation summaries,
`metrics.json`, and evidence/reasoning details under `RUN_DIR/eval_pipeline/`.

Add `--eval-only` to a shell command to evaluate existing predictions. For individual
stages, use `bench/evaluation/evaluate.py` or `evaluate_grouping.py` with `--stages`.
Stages 1 and 2 do not call a model API; Stage 3 uses the judge API. Run Stages
2, 3, and 4 together to calculate Verified F1.

## Usage notes

- Use a new result directory after changing data, prompts, or model settings.
- Check failed requests, parse errors, and sample coverage alongside scores.
- Use JSON booleans (`true` / `false`), not strings such as `"false"`.
- Evidence checks use source text and may include content beyond the inference truncation limit.
