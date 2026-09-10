# Inputs

Download the three JSON files from
[ZitaGo/NovGauge](https://huggingface.co/datasets/ZitaGo/NovGauge/tree/main/data)
into `data/`. Add local paper content without changing IDs, titles, paper order,
labels, or evaluated dimensions.

## Paper content

Each paper record needs `markdown_path` for full-text evaluation and `abstract`
for abstract-based modes:

```json
{
  "paper_id": "synthetic-a",
  "title": "Synthetic Paper A",
  "markdown_path": "examples/markdown/paper_a.md",
  "abstract": "An abstract supplied by the user."
}
```

Paths may be absolute or relative to the repository root. Use readable Markdown
with headings such as `## Introduction` and `## References`. Retain the dataset's
bibliographic metadata, including `authors` when available.

## Labels and examples

- **Pairwise:** `paper_a`, `paper_b`, and `labels`. Values are `1` (similar),
  `0` (dissimilar), or `null` (unevaluated).
  See [positives](../examples/positives.json) and [negatives](../examples/negatives.json).
- **Grouping:** `papers`, `ground_truth`, and `eval_dims`. Group indices are
  one-based and refer to the order in `papers`; only `eval_dims` are scored.
  See [grouping](../examples/grouping.json).

## Content modes

| Mode | Content used |
| --- | --- |
| `abstract` | The JSON `abstract` field |
| `abstract+intro` | Abstract plus the Markdown introduction, or cached content |
| `full` | Markdown before the references section |

Missing Markdown may fall back to the abstract. Check local files before running.
The optional cache is `bench/output/cache/paper_content_cache_human.json`, keyed
by `markdown_path`; remove stale entries when changing paper content.
Tokenization uses `cl100k_base`, which may download its encoding on first use.
