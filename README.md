# NovGauge

Evaluate paper similarity across **task**, **problem**, and **method**, using
pairwise classification and multi-paper grouping. The framework checks model
predictions, evidence, and reasoning, and reports similarity metrics and Verified F1.

[Dataset on Hugging Face](https://huggingface.co/datasets/ZitaGo/NovGauge)
· [Input format](docs/INPUTS.md)
· [Evaluation guide](docs/EVALUATION.md)

## Quick start

Requires **Python 3.10+**, Bash, and an OpenAI-compatible model API.

### 1. Install

```bash
git clone https://github.com/Zita-Go/NovGauge.git
cd NovGauge
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` to set `NOVGAUGE_BASE_URL`, `NOVGAUGE_API_KEY`, `EVAL_MODEL`, and
`JUDGE_MODEL` for your provider. Then load the settings:

```bash
set -a
source .env
set +a
```

### 3. Run the examples

The examples contain synthetic paper content. These commands run the full
evaluation pipeline and make API calls that may incur charges.

```bash
# Pairwise classification
POS_INPUT=examples/positives.json NEG_INPUT=examples/negatives.json \
RUN_DIR=results/pairwise-example bash bench/run_eval.sh

# Multi-paper grouping
INPUT=examples/grouping.json RUN_DIR=results/grouping-example \
bash bench/run_grouping_eval.sh
```

Predictions and evaluation results are saved under `results/`.

## Run the benchmark

Download `positives.json`, `negatives.json`, and `grouping.json` from
[Hugging Face](https://huggingface.co/datasets/ZitaGo/NovGauge/tree/main/data)
into `data/`. Add a local `markdown_path` to each paper record; abstract-based
modes also require an `abstract` field. See the [input guide](docs/INPUTS.md).
Paper acquisition and PDF-to-Markdown conversion are outside this framework.

```bash
bash bench/run_eval.sh
bash bench/run_grouping_eval.sh
```

Both tasks use the same inference defaults with thinking enabled.
The default content mode is `full`; set `CONTENT_MODE` to `abstract` or
`abstract+intro` to change it. See the [evaluation guide](docs/EVALUATION.md)
for parameters, individual stages, and output formats.

## License

Code: [MIT](LICENSE). Dataset: [CC BY 4.0](https://huggingface.co/datasets/ZitaGo/NovGauge).
