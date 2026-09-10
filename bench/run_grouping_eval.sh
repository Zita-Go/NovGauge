#!/usr/bin/env bash
# Run grouping inference and all four evaluation stages. Use --help for options.

set -euo pipefail
cd "$(dirname "$0")/.."          # Always run from the repository root.

# User configuration

# Evaluation model (used by run_grouping_bench)
EVAL_MODEL="${EVAL_MODEL:-openai/gpt-5.4}"
EVAL_BASE_URL="${EVAL_BASE_URL:-${NOVGAUGE_BASE_URL:-}}"
EVAL_API_KEY="${EVAL_API_KEY:-${NOVGAUGE_API_KEY:-}}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.6}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-1}"
EVAL_THINKING_BUDGET="${EVAL_THINKING_BUDGET:-8000}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
EVAL_MAX_FULL_TOKENS="${EVAL_MAX_FULL_TOKENS:-16000}"
EVAL_MAX_INTRO_TOKENS="${EVAL_MAX_INTRO_TOKENS:-4000}"
EVAL_MAX_ABSTRACT_TOKENS="${EVAL_MAX_ABSTRACT_TOKENS:-1000}"

# Judge model (used by check_reasoning_grouping / Stage 3)
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-${NOVGAUGE_BASE_URL:-}}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${NOVGAUGE_API_KEY:-}}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"

# Content mode (abstract | abstract+intro | full)
CONTENT_MODE="${CONTENT_MODE:-full}"

# Evaluation dimensions
DIMS="${DIMS:-task problem method}"

# Number of workers
WORKERS="${WORKERS:-8}"

# Input data
INPUT="${INPUT:-data/grouping.json}"

# Derived paths (usually no changes needed)

MODEL_TAG="${EVAL_MODEL//\//_}"
RUN_DIR="${RUN_DIR:-results/grouping_${MODEL_TAG}_${CONTENT_MODE}}"
RESULTS_FILE="${RUN_DIR}/${MODEL_TAG}_${CONTENT_MODE}.json"
EVAL_DIR="${RUN_DIR}/eval_pipeline"

# Argument parsing

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
if [ ! -f "${INPUT}" ] || [ ! -r "${INPUT}" ]; then
  printf 'ERROR: INPUT is not a readable file: %s\n' "${INPUT}" >&2
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

# Step 1: run_grouping_bench

if [ "$EVAL_ONLY" = false ]; then
  echo "============================================================"
  echo "  Step 1: run_grouping_bench  [model=${EVAL_MODEL}, mode=${CONTENT_MODE}]"
  echo "============================================================"

  mkdir -p "${RUN_DIR}"

  echo ""
  echo "  Results → ${RESULTS_FILE}"
  python bench/run_grouping_bench.py \
    --input              "${INPUT}" \
    --models             "${EVAL_MODEL}" \
    --content-modes      "${CONTENT_MODE}" \
    --dims               ${DIMS} \
    --output-dir         "${RUN_DIR}" \
    --base-url           "${EVAL_BASE_URL}" \
    --api-key            "${EVAL_API_KEY}" \
    --workers            "${WORKERS}" \
    --temperature        "${EVAL_TEMPERATURE}" \
    --n-samples          "${EVAL_N_SAMPLES}" \
    --thinking-budget    "${EVAL_THINKING_BUDGET}" \
    --max-tokens         "${EVAL_MAX_TOKENS}" \
    --max-tokens-per-paper-full     "${EVAL_MAX_FULL_TOKENS}" \
    --max-tokens-per-paper-intro    "${EVAL_MAX_INTRO_TOKENS}" \
    --max-tokens-per-paper-abstract "${EVAL_MAX_ABSTRACT_TOKENS}"
else
  echo "  [--eval-only] Skipping Step 1; using existing results:"
  echo "    RESULTS: ${RESULTS_FILE}"

  if [ ! -f "${RESULTS_FILE}" ]; then
    echo "  ERROR: ${RESULTS_FILE} does not exist; run Step 1 first or check the path"
    exit 1
  fi
fi

# Step 2: Four-stage evaluation

echo ""
echo "============================================================"
echo "  Step 2: evaluate_grouping  [judge=${JUDGE_MODEL}]"
echo "============================================================"

mkdir -p "${EVAL_DIR}"

python bench/evaluation/evaluate_grouping.py \
  --results            "${RESULTS_FILE}" \
  --input              "${INPUT}" \
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
