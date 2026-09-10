#!/usr/bin/env bash
# ============================================================
# NovGauge end-to-end pairwise evaluation
#
# Workflow:
#   Step 1  run_bench      — Predict pairwise similarity with an LLM (writes POS/NEG_RESULTS)
#   Step 2  evaluate.py    — Four-stage evaluation
#             Stage 1: Accuracy / Precision / Recall / F1
#             Stage 2: Hallucination rate (reported separately for TP and FP)
#             Stage 3: Mismatch rate (pred=1 samples with no hallucinated evidence)
#             Stage 4: Verified F1 (both evidence and reasoning pass verification)
#
# Usage:
#   bash bench/run_eval.sh            # Full workflow (Step 1 + Step 2)
#   bash bench/run_eval.sh --eval-only  # Skip Step 1 and run Step 2
#                                       # (use after run_bench has completed)
# ============================================================

set -euo pipefail
cd "$(dirname "$0")/.."          # Always run from the repository root.

# ============================================================
# User configuration
# ============================================================

# --- Evaluation model (used by run_bench)---
EVAL_MODEL="${EVAL_MODEL:-openai/gpt-5.4}"
EVAL_BASE_URL="${EVAL_BASE_URL:-${NOVGAUGE_BASE_URL:-}}"
EVAL_API_KEY="${EVAL_API_KEY:-${NOVGAUGE_API_KEY:-}}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"         # Sampling temperature (0=greedy; positive values add randomness)
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"             # Independent samples per task; with >1, compute F1 per sample and report mean +/- std
EVAL_THINKING_BUDGET="${EVAL_THINKING_BUDGET:-8000}"    # Reasoning budget_tokens (0=disabled, >0=enabled; suggested value: 8000)
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"           # LLM output token limit
EVAL_MAX_FULL_TOKENS="${EVAL_MAX_FULL_TOKENS:-16000}"     # Per-paper content token limit in full mode (total input is approximately twice this limit plus the template)
EVAL_MAX_INTRO_TOKENS="${EVAL_MAX_INTRO_TOKENS:-4000}"     # Per-paper content token limit in abstract+intro mode
EVAL_MAX_ABSTRACT_TOKENS="${EVAL_MAX_ABSTRACT_TOKENS:-1000}"  # Per-paper content token limit in abstract mode

# --- Judge model (used by check_reasoning / Stage 3)---
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${NOVGAUGE_BASE_URL:-}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${NOVGAUGE_API_KEY:-}}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"          # Suggested judge temperature: 0 for deterministic output
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"         # Judge output token limit

# --- Content mode (abstract | abstract+intro | full)---
CONTENT_MODE="${CONTENT_MODE:-full}"

# --- Evaluation dimensions ---
DIMS="${DIMS:-task problem method}"

# --- Number of workers ---
WORKERS="${WORKERS:-8}"

# --- Input data ---
POS_INPUT="${POS_INPUT:-data/positives.json}"
NEG_INPUT="${NEG_INPUT:-data/negatives.json}"

# ============================================================
# Derived paths (usually no changes needed)
# ============================================================

MODEL_TAG="${EVAL_MODEL//\//_}"   # openai/gpt-5.4 → openai_gpt-5.4
RUN_DIR="${RUN_DIR:-results/${MODEL_TAG}_${CONTENT_MODE}}"
EVAL_DIR="${RUN_DIR}/eval_pipeline"

# Result files written by run_bench (produced in Step 1, read in Step 2)
POS_RESULTS="${RUN_DIR}/positives/${MODEL_TAG}/${CONTENT_MODE//+/_}.json"
NEG_RESULTS="${RUN_DIR}/negatives/${MODEL_TAG}/${CONTENT_MODE//+/_}.json"

# ============================================================
# Argument parsing
# ============================================================

EVAL_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --eval-only) EVAL_ONLY=true ;;
    -h|--help)
      printf 'Usage: bash %s [--eval-only]\n' "${BASH_SOURCE[0]}"
      printf 'Configure model, endpoint, credentials, input files and content mode through environment variables.\n'
      exit 0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# Check configuration before inference can incur API cost.
case "$CONTENT_MODE" in
  abstract|abstract+intro|full) ;;
  *) printf 'ERROR: CONTENT_MODE must be abstract, abstract+intro, or full.\n' >&2; exit 2 ;;
esac
if [ ! -f "${POS_INPUT}" ] || [ ! -r "${POS_INPUT}" ]; then
  printf 'ERROR: POS_INPUT is not a readable file: %s\n' "${POS_INPUT}" >&2
  exit 2
fi
if [ ! -f "${NEG_INPUT}" ] || [ ! -r "${NEG_INPUT}" ]; then
  printf 'ERROR: NEG_INPUT is not a readable file: %s\n' "${NEG_INPUT}" >&2
  exit 2
fi
required_config=(JUDGE_BASE_URL JUDGE_API_KEY)
if [ "$EVAL_ONLY" = false ]; then
  required_config+=(EVAL_BASE_URL EVAL_API_KEY)
fi
for config_name in "${required_config[@]}"; do
  if [ -z "${!config_name}" ]; then
    printf 'ERROR: Set %s or the corresponding NOVGAUGE_BASE_URL / NOVGAUGE_API_KEY.\n' "$config_name" >&2
    exit 2
  fi
done

# ============================================================
# Step 1: run_bench (positives and negatives)
# ============================================================

if [ "$EVAL_ONLY" = false ]; then
  echo "============================================================"
  echo "  Step 1: run_bench  [model=${EVAL_MODEL}, mode=${CONTENT_MODE}]"
  echo "============================================================"

  mkdir -p "$(dirname "$POS_RESULTS")"
  mkdir -p "$(dirname "$NEG_RESULTS")"

  echo ""
  echo "  [1a] Positives → ${POS_RESULTS}"
  python bench/evaluation/run_bench.py \
    --input            "${POS_INPUT}" \
    --models           "${EVAL_MODEL}" \
    --content-modes    "${CONTENT_MODE}" \
    --dims             ${DIMS} \
    --output-dir       "${RUN_DIR}/positives" \
    --base-url         "${EVAL_BASE_URL}" \
    --api-key          "${EVAL_API_KEY}" \
    --workers          "${WORKERS}" \
    --temperature      "${EVAL_TEMPERATURE}" \
    --n-samples        "${EVAL_N_SAMPLES}" \
    --thinking-budget  "${EVAL_THINKING_BUDGET}" \
    --max-tokens           "${EVAL_MAX_TOKENS}" \
    --max-full-tokens      "${EVAL_MAX_FULL_TOKENS}" \
    --max-intro-tokens     "${EVAL_MAX_INTRO_TOKENS}" \
    --max-abstract-tokens  "${EVAL_MAX_ABSTRACT_TOKENS}"

  echo ""
  echo "  [1b] Negatives → ${NEG_RESULTS}"
  python bench/evaluation/run_bench.py \
    --input            "${NEG_INPUT}" \
    --models           "${EVAL_MODEL}" \
    --content-modes    "${CONTENT_MODE}" \
    --dims             ${DIMS} \
    --output-dir       "${RUN_DIR}/negatives" \
    --base-url         "${EVAL_BASE_URL}" \
    --api-key          "${EVAL_API_KEY}" \
    --workers          "${WORKERS}" \
    --temperature      "${EVAL_TEMPERATURE}" \
    --n-samples        "${EVAL_N_SAMPLES}" \
    --thinking-budget  "${EVAL_THINKING_BUDGET}" \
    --max-tokens           "${EVAL_MAX_TOKENS}" \
    --max-full-tokens      "${EVAL_MAX_FULL_TOKENS}" \
    --max-intro-tokens     "${EVAL_MAX_INTRO_TOKENS}" \
    --max-abstract-tokens  "${EVAL_MAX_ABSTRACT_TOKENS}"
else
  echo "  [--eval-only] Skipping Step 1; using existing results:"
  echo "    POS: ${POS_RESULTS}"
  echo "    NEG: ${NEG_RESULTS}"

  if [ ! -f "${POS_RESULTS}" ]; then
    echo "  ERROR: ${POS_RESULTS} does not exist; run Step 1 first or check the path"
    exit 1
  fi
  if [ ! -f "${NEG_RESULTS}" ]; then
    echo "  ERROR: ${NEG_RESULTS} does not exist; run Step 1 first or check the path"
    exit 1
  fi
fi

# ============================================================
# Step 2: Four-stage evaluation
# ============================================================

echo ""
echo "============================================================"
echo "  Step 2: evaluate  [judge=${JUDGE_MODEL}]"
echo "============================================================"

mkdir -p "${EVAL_DIR}"

python bench/evaluation/evaluate.py \
  --pos-results        "${POS_RESULTS}" \
  --neg-results        "${NEG_RESULTS}" \
  --pos-inputs         "${POS_INPUT}" \
  --neg-inputs         "${NEG_INPUT}" \
  --mode               "${CONTENT_MODE}" \
  --base-url           "${JUDGE_BASE_URL}" \
  --api-key            "${JUDGE_API_KEY}" \
  --model              "${JUDGE_MODEL}" \
  --judge-temperature  "${JUDGE_TEMPERATURE}" \
  --judge-max-tokens   "${JUDGE_MAX_TOKENS}" \
  --output-dir         "${EVAL_DIR}" \
  --workers            "${WORKERS}" \
  --skip-existing

echo ""
echo "============================================================"
echo "  Done."
echo "  Results in: ${EVAL_DIR}"
echo "============================================================"
