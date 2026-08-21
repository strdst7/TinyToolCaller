---
license: cc-by-4.0
task_categories:
  - text-generation
language:
  - en
pretty_name: TinyToolCaller
size_categories:
  - 1K<n<10K
tags:
  - function-calling
  - tool-calling
  - qlora
  - qwen2.5
---

# TinyToolCaller Dataset

`strdst77/TinyToolCaller` is a derived supervised fine-tuning dataset for a narrow function-calling task: map a user request and available tool schemas to one JSON object containing `name` and `arguments`.

| Property | Value |
| --- | --- |
| Source | `Salesforce/xlam-function-calling-60k` (APIGen; gated) |
| Sampling | `shuffle(seed=42)` then first 5,200 rows |
| Splits | 5,000 train / 200 development |
| Row fields | `query`, `tools`, `answers` |
| Companion code | [strdst7/TinyToolCaller](https://github.com/strdst7/TinyToolCaller) |

## Data use and limitations

- The source dataset has access conditions. Obtain access and comply with its terms before recreating or redistributing the subset.
- Preserve source attribution and verify the appropriate derived-data license before release.
- This dataset supports single-call structured prediction. It does not demonstrate multi-turn planning, tool execution safety, or general function-calling capability.
- The released subset must be profiled for tool concentration, prompt length, truncation, and train/development coverage before using it to make generalization claims.

## Reproduction

```bash
export HF_TOKEN=<accepted-source-dataset-token>
python scripts/profile_tool_distribution.py
python scripts/dataset_stats.py
```

The complete methodology, implementation details, evaluation boundaries, and deployment guidance are in the repository [publication](README.md).
