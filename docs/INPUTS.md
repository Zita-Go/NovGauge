# Input contract

Inputs are UTF-8 JSON arrays. The original evaluator schemas are retained.
Dataset annotations and local paper content are supplied by the user. Before
running evaluation, provide `markdown_path` and `abstract` in each paper record
as required by the selected mode. This repository does not assemble datasets or
generate annotations.

## Paper metadata and content

| Field | Meaning |
| --- | --- |
| `paper_id` | Stable paper identifier retained from the benchmark |
| `title` | Paper title included in prompts and evidence checking |
| `authors` | Author-name list included in grouping prompts; an empty list displays as unknown |
| `arxiv_id`, `doi`, `forum_id`, `version`, `source_url` | Optional provenance retained from the dataset |
| `markdown_path` | Local Markdown path, relative to this repository root or absolute |
| `abstract` | Locally supplied abstract; required for useful abstract-based evaluation |

Use actual paper-version identifiers in real runs. Synthetic example IDs do not
refer to real publications. Keep titles and paper ordering exactly aligned with
the annotation files.

## Pairwise labels

```json
[
  {
    "pair_id": "synthetic-pair-1",
    "paper_a": {"paper_id": "synthetic-a", "title": "Synthetic Paper A"},
    "paper_b": {"paper_id": "synthetic-b", "title": "Synthetic Paper B"},
    "labels": {"task": 1, "problem": 1, "method": 0}
  }
]
```

`1` means similar, `0` means dissimilar, and a missing dimension or `null` means
unevaluated. The original inference runner also supports `submitted_paper` /
`prior_work` and `similar_dimensions`. A nonempty `similar_dimensions` list takes
precedence over `labels` and marks only those dimensions as positive; dimensions
outside that list are not automatically negative. Do not invent missing labels.

For all-stage pairwise evaluation use `paper_a` / `paper_b` or
`submitted_paper` / `prior_work`. The inference-only `negative_paper` alias is not
understood consistently by the original evidence checker; normalize a local
input to a supported pair without changing labels.

## Grouping labels

```json
[
  {
    "item_id": "synthetic-group-1",
    "source_survey": "synthetic-source",
    "n_papers": 3,
    "papers": [
      {"paper_index": 1, "paper_id": "synthetic-a", "title": "Synthetic Paper A"},
      {"paper_index": 2, "paper_id": "synthetic-b", "title": "Synthetic Paper B"},
      {"paper_index": 3, "paper_id": "synthetic-c", "title": "Synthetic Paper C"}
    ],
    "ground_truth": {
      "task_groups": [[1, 2]],
      "problem_groups": [[1, 2]],
      "method_groups": []
    },
    "eval_dims": ["task", "problem"]
  }
]
```

Indices are one-based and refer to the existing paper order. Preserve `eval_dims`
and empty groups exactly; unevaluated dimensions must not be relabeled.

## Local content fields

A paper record with local content has the following form:

```json
{
  "paper_id": "synthetic-a",
  "title": "Synthetic Paper A",
  "markdown_path": "examples/markdown/paper_a.md",
  "abstract": "An abstract supplied by the user."
}
```

Use the paper records inside the pairwise or grouping schemas above. Ensure the
local files are readable and correspond to the same paper versions as the
benchmark. The runnable examples already contain these fields.

## Markdown and caching

Markdown should have readable text and ATX headings (`#`, `##`, etc.). Pairwise
and grouping heading handling differ in the original code and remain separate.
The pairwise and grouping preprocessing rules are retained from the original
evaluation implementation.

- `abstract`: uses the JSON `abstract` field.
- `abstract+intro`: uses the optional local cache first, otherwise the original
  Markdown introduction extraction and abstract fallback.
- `full`: reads Markdown, applies the original reference-removal rules, then
  truncates for inference.

The optional cache remains at
`bench/output/cache/paper_content_cache_human.json`, keyed by the exact
`markdown_path` string. A value may contain `abstract` and `abstract+intro` text.
No cache or real text is distributed. Starting without a cache selects the
original fallback path; this may differ from historical runs that used a cache.

Missing Markdown can fall back to the abstract in the runners. Check local files
before running evaluation to avoid unintended fallback. Evidence checks retain their original
content loading and normalization, including their token-limit behavior.

`tiktoken` uses `cl100k_base`. Its first use may need to download the encoding;
cache it in advance when operating offline. The original character-based
fallback when the tokenizer cannot load is retained. Record content hashes,
cache usage, input versions, tokenizer, model IDs and all evaluation settings
when comparing scores.
