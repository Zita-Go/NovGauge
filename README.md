# NovGauge evaluation framework

NovGauge evaluates paper similarity along **task**, **problem**, and **method**
dimensions, using pairwise classification and multi-paper grouping. This
repository starts with user-provided paper Markdown and metadata.

The benchmark annotations have a separate Hugging Face dataset repository:
[ZitaGo/NovGauge](https://huggingface.co/datasets/ZitaGo/NovGauge).
This repository currently includes only synthetic examples, not the benchmark
annotations or real paper content.

The original experiments used an internal PDF-to-Markdown service. Paper
download, PDF conversion, the internal service and its credentials are outside
the release scope. Different Markdown extraction results can change scores.

## Installation

Run commands from this repository's root with Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`bench/evaluation` contains the pairwise runner and evaluation stages. The grouping
runner is `bench/run_grouping_bench.py`. `bench/llm_client.py` provides the HTTP
client used by the reasoning judges.

## Prepare local inputs

Download the annotations from
[Hugging Face](https://huggingface.co/datasets/ZitaGo/NovGauge) and provide your
own Markdown. The expected schema is documented in [docs/INPUTS.md](docs/INPUTS.md).
Paths in `markdown_path` are relative to this repository root, or absolute.
The `abstract` field must be supplied locally for abstract-based modes; the
framework does not automatically read the abstract from Markdown.

The dataset provides `data/positives.json`, `data/negatives.json`, and
`data/grouping.json`. Download them into this repository's local `data/` directory
and add the content fields described below. To select a fixed dataset version,
use the Hugging Face commit
`3b4d9589ca72499b30cf1fb766f143b74e6dbe89`.

Each paper record in your local evaluation JSON must point to its Markdown
through `markdown_path`. Supply `abstract` for abstract-based modes. Preserve
the benchmark sample IDs, paper ordering, labels, and evaluated dimensions.
Dataset construction, annotation and input assembly are outside this framework.

`examples/positives.json`, `examples/negatives.json`, and
`examples/grouping.json` are already populated with synthetic local content
references. They are suitable for checking the interface, not model quality.

## Configure the API

```bash
cp .env.example .env
# Edit .env with your endpoint, credentials, and provider model identifiers.
set -a
source .env
set +a
```

No credentials or provider address are embedded in the Python code.
The inference runners and integrated evaluators accept `NOVGAUGE_API_KEY` and
`NOVGAUGE_BASE_URL`; explicit `--api-key` and `--base-url` arguments still work.
The standalone reasoning-check scripts retain their original required CLI flags.
Environment settings must be exported explicitly; the framework does not search
parent directories for `.env` files.

## Run the complete evaluation

The shell entry points run inference followed by all four evaluation stages:
similarity metrics, evidence checking, reasoning checking, and verified F1.
They make paid API calls to the endpoints you configure.

```bash
POS_INPUT=examples/positives.json \
NEG_INPUT=examples/negatives.json \
RUN_DIR=results/pairwise-example \
bash bench/run_eval.sh

INPUT=examples/grouping.json \
RUN_DIR=results/grouping-example \
bash bench/run_grouping_eval.sh
```

For benchmark runs, replace these paths with your locally prepared dataset
files. Supported content modes are `abstract`, `abstract+intro`, and `full`.
Set `CONTENT_MODE` to choose a mode. `--eval-only` skips inference and reads
the corresponding existing result files.

The default inputs are `data/positives.json`, `data/negatives.json`, and
`data/grouping.json`. Generated outputs go under `results/`. The shell entry
points check input-file readability, content mode and both API configurations
before starting inference. `bash bench/run_eval.sh --help` and
`bash bench/run_grouping_eval.sh --help` show their usage.

The shell scripts retain the original model and numerical defaults. Override
settings such as `EVAL_MODEL`, `JUDGE_MODEL`, `EVAL_N_SAMPLES`,
`EVAL_TEMPERATURE`, `EVAL_THINKING_BUDGET`, and `WORKERS` through environment
variables. Python entry-point defaults differ from the shell presets; use the
same explicit parameters when comparing experiments. Provider-specific model
aliases, streaming behavior, and thinking parameters are preserved from the
source implementation and may need a compatible endpoint.

## Use Python entry points directly

```bash
python bench/evaluation/run_bench.py \
  --input examples/positives.json \
  --models "$EVAL_MODEL" \
  --content-modes full \
  --output-dir results/positives

python bench/run_grouping_bench.py \
  --input examples/grouping.json \
  --models "$EVAL_MODEL" \
  --content-modes full \
  --output-dir results/grouping
```

To score saved pairwise predictions without making API calls:

```bash
python bench/evaluation/evaluate.py \
  --pos-results results/positives/MODEL_SLUG/full.json \
  --pos-inputs examples/positives.json \
  --mode full \
  --stages 1 2 \
  --output-dir results/scores
```

Stage 3 needs the judge API. Verified F1 (Stage 4) should be interpreted only
with the corresponding Stage 2 and Stage 3 results. Do not reuse results across
different Markdown, prompts, models, datasets, or configurations under the same
output directory; the original resume behavior is preserved.
See [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions, the different
entry-point presets, output filenames and reproducibility limits.

## License

The evaluation code is licensed under the [MIT License](LICENSE).
The [HF dataset](https://huggingface.co/datasets/ZitaGo/NovGauge) is licensed
separately under CC BY 4.0.
Third-party paper content is not relicensed by this repository.
