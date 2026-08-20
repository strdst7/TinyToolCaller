---
license: apache-2.0
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
  - instruction-tuning
  - apigen
---

# TinyToolCaller (Dataset Card)

**Derived supervised fine-tuning subset for reliable function calling.**

| | |
| --- | --- |
| **Dataset** | `strdst77/TinyToolCaller` |
| **Source** | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC-BY-4.0) |
| **Sampling** | `shuffle(seed=42)` → 5,200 examples → 5,000 train / 200 validation |
| **Splits** | `train.parquet` (5,000), `validation.parquet` (200) |
| **Format** | One example = `query` + serialized `tools` + ground-truth `answers` (ChatML-formatted at training time) |
| **Task** | Structured tool calling: map request + tool schemas → `{"name": ..., "arguments": {...}}` |
| **Companion code** | [`strdst7/TinyToolCaller`](https://github.com/strdst7/TinyToolCaller) |
| **License** | Apache-2.0 (derived subset); source dataset CC-BY-4.0 — see §29 |

This card is the publication for the TinyToolCaller project. The full write-up follows.

---

# TinyToolCaller

### QLoRA Fine-Tuning of a 1.5B LLM for Reliable Function Calling

| | |
| --- | --- |
| **Project** | TinyToolCaller |
| **Base model** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| **Task** | Structured function / tool calling (JSON tool selection + argument construction) |
| **Method** | QLoRA (4-bit NF4) + LoRA/PEFT, supervised fine-tuning via TRL `SFTTrainer` |
| **Source dataset** | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (gated — authentication required) |
| **Project dataset** | [`strdst77/TinyToolCaller`](https://huggingface.co/datasets/strdst77/TinyToolCaller) (Hugging Face) |
| **Code** | [`strdst7/TinyToolCaller`](https://github.com/strdst7/TinyToolCaller) (GitHub) |
| **Tracking** | Weights & Biases |
| **Status** | Research / applied LLM engineering study — pre-publication draft |

> **Abstract.** TinyToolCaller is an applied study of parameter-efficient specialization: it takes a small open-weight instruction model, `Qwen2.5-1.5B-Instruct`, and fine-tunes it with QLoRA on a 5,000-example supervised subset of the Salesforce xLAM function-calling dataset so that, given a natural-language request and a set of tool schemas, the model emits a single, machine-readable tool call — `{"name": ..., "arguments": {...}}` — with no markdown, commentary, or extraneous text. Against the project's 200-example evaluation split, the fine-tuned model improves **JSON validity from 78.5% → 98.0%**, **tool-name accuracy from 65.0% → 92.5%**, and **argument exact-match from 42.0% → 84.0%**, while a 50-example GSM8K retention check moves 52.0% → 50.0% (within sampling noise). The work's contribution is **not** a claim of state-of-the-art tool calling; it is a transparent, reproducible ablation isolating how much lift QLoRA alone provides over an unmodified 1.5B base model, together with an explicit accounting of the evaluation's limitations and a deterministic-runtime design in which the LLM produces structured intent while application code owns validation, authorization, and execution.

---

> **⚠️ Before quoting the headline numbers.** The results in §19–§21 are **in-sample**: the 200-example split is also the development/evaluation set (no independent held-out test set), the GSM8K check uses 50 examples, and the **tool-distribution profile of the 5,200-example subset has not yet been measured** (§8.1). Until §8.1's three quantities — (a) unique tool count, (b) top-10 tool frequency, (c) train/validation distribution match — are filled in, the tool-name-accuracy figures cannot be separated from possible selection skew. Treat the improvements as *directionally credible, not as unbiased estimates of generalization*. See §20, §22, §27.

---

# 1. Executive Summary

TinyToolCaller specializes a 1.5B-parameter open-weight model for a deliberately narrow capability: **reliable function calling**. The objective is not general competence but a specific contract — map

```text
natural-language request + available tool schemas  ⟶  { "name": ..., "arguments": {...} }
```

and nothing else. The project uses **QLoRA** so the base model can be specialized without full-model optimization, then measures the result with identical metrics before and after fine-tuning.

Using 5,000 training examples and a 200-example evaluation split (sampled with seed `42` from the 60K source), the experiment reports:

| Metric | Base model | TinyToolCaller | Change |
| --- | ---: | ---: | ---: |
| JSON validity (extraction-based) | 78.5% | **98.0%** | +19.5 pp |
| Tool-name accuracy | 65.0% | **92.5%** | +27.5 pp |
| Argument exact match | 42.0% | **84.0%** | +42.0 pp |
| GSM8K (50-example retention check) | 52.0% | **50.0%** | −2.0 pp |

These are **experimental results from this project** — in-sample, with the caveats of §20–§22 — and are not a claim that TinyToolCaller outperforms larger or production-grade function-calling models.

# 2. Problem Statement

LLMs increasingly act as the interface between natural-language users and software systems. A conversational answer —

> "The weather in Tokyo is likely to be sunny."

— is insufficient when the model must operate a real API. A tool-using system needs a structured, executable representation:

```json
{ "name": "get_weather", "arguments": { "location": "Tokyo" } }
```

A language model can fail this task at **seven distinct levels**:

1. produce invalid JSON;
2. wrap JSON in markdown fences;
3. append explanatory text;
4. select the wrong tool;
5. omit required arguments;
6. generate incorrect argument values;
7. invent arguments absent from the schema.

TinyToolCaller addresses these through supervised specialization. The output contract is a single object with exactly two fields — `name` and `arguments` — and no additional commentary or formatting.

# 3. Project Objective

Six objectives define the project:

1. **Dataset preparation** — transform a public function-calling dataset into instruction–response examples for supervised fine-tuning.
2. **Baseline measurement** — measure the unmodified base model's reliability on the task before fine-tuning.
3. **Parameter-efficient fine-tuning** — apply QLoRA to specialize the model without updating the full base.
4. **Post-fine-tuning evaluation** — compare fine-tuned vs. base model with identical metrics.
5. **Capability retention** — check for degradation on a general reasoning task (GSM8K).
6. **Reproducibility** — publish code, derived dataset, methodology, results, and artifacts for inspection and reproduction.

# 4. Intended Audience and Use Case

## Intended audience

ML/LLM engineers, AI application and agent developers, researchers exploring parameter-efficient fine-tuning, students learning practical LLM fine-tuning, developers working on function calling, engineers building low-cost inference systems, and practitioners evaluating small language models for specialized tasks. It is most useful to readers who want to understand **how a small open-weight model can be specialized for a narrowly defined operational capability**.

## Intended use case

The intended use case is **structured tool selection**: a downstream application supplies

```text
User Request  +  Available Tool Schemas
```

and TinyToolCaller returns

```text
Tool Name  +  Tool Arguments
```

which the application can then validate and execute:

```text
LLM                                     Deterministic application
│  Understand request                   │
├─ Select tool                          │  Validate
└─ Construct arguments                  │  Authorize
        │                               │  Execute
        ▼                               ▼
  Structured JSON ─────────────────▶  Tool execution
```

This project does **not** claim the LLM should directly execute arbitrary external functions.

# 5. Real-World Applications

Reliable function calling underpins many AI systems:

| Scenario | Request | Expected call |
| --- | --- | --- |
| Personal assistant | "Add a dentist appointment tomorrow at 3 PM." | `create_calendar_event(...)` |
| Customer support | "What's the status of order 12345?" | `get_order_status(...)` |
| Enterprise search | "Find all invoices from Vendor X this quarter." | `search_financial_records(...)` |
| Weather / info systems | "What's the weather in Tokyo?" | `get_weather(...)` |
| Database assistant | "Show me customers who haven't purchased in 90 days." | `query_customer_database(...)` |
| Workflow automation | "Create a support ticket and assign it to infra." | `create_ticket(...)` → `assign_ticket(...)` |

In each case the model's job is to translate human intent into a machine-readable representation that deterministic software can process.

# 6. Dataset Sources and Collection

## Context: how small-model function calling is usually done

1. **Base-model prompting** — rely on existing instruction-following without adaptation (this project's baseline measures exactly this).
2. **Full fine-tuning** — update all parameters on function-calling data; effective but expensive for iterative single-GPU work.
3. **Parameter-efficient fine-tuning (LoRA/QLoRA)** — the approach here, and increasingly the default for small-model specialization because it fits single-GPU workflows.

Existing models (Qwen's function-calling variants, Hermes function-calling models, Gorilla) already target this capability at similar or larger scale. **TinyToolCaller does not claim to outperform them** — it isolates and measures how much lift QLoRA alone provides over an unmodified 1.5B base model with a fixed, reproducible recipe. That is the gap it fills: a transparent, small-scale ablation rather than a new state-of-the-art model.

## Source dataset

TinyToolCaller uses [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), part of the APIGen function-calling work. Per the dataset card, APIGen generates verifiable, diverse function-calling data: 60,000 examples covering 3,673 executable APIs across 21 categories, produced with hierarchical verification (format checking, actual function execution, semantic verification). The card reports human evaluation over 600 sampled examples with a correctness rate above 95%, while acknowledging remaining minor issues.

> **Access note.** This dataset is **gated** on the Hugging Face Hub: loading it requires a logged-in account that has accepted the dataset terms (`HF_TOKEN`). This is stated explicitly here because it affects reproducibility (§30).

# 7. Dataset Collection and Sampling

```text
Source dataset
    ↓  shuffle(seed = 42)
Select 5,200 examples
    ↓
5,000 training  +  200 validation
```

A 5,200-example subset (seed `42`) is used instead of the full 60K for lower compute requirements, faster experimentation, reduced cost, and easier single-GPU iteration. The trade-off is reduced coverage of the source's 3,673 APIs — a trade-off §8.1 makes measurable.

# 8. Dataset Description

Each example pairs a natural-language query with the tools available to answer it and the expected call:

```json
{
  "query": "What's the weather in Tokyo?",
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather",
      "parameters": { "location": "string", "unit": "celsius|fahrenheit" }
    }
  ],
  "answers": [
    { "name": "get_weather", "arguments": { "location": "Tokyo", "unit": "celsius" } }
  ]
}
```

The model is therefore learning a mapping — *request + candidate tools → tool selection + argument construction* — not merely an answer.

Because the subset is a random draw from a source spanning 3,673 APIs across 21 categories, its **tool and category distribution has not historically been profiled**. That is a real gap: a subset skewed toward a handful of high-frequency tools would inflate tool-accuracy figures relative to a balanced sample. §8.1 defines the required measurement and the tooling to produce it.

## 8.1 Tool-Distribution Profiling *(required before quoting results)*

The single most important open measurement for this project is the tool distribution of the 5,200-example subset. Three quantities must be reported:

- **(a) Unique tool count** — the number of distinct ground-truth tool names (`name`) appearing in the 5,200-example subset, to characterize coverage of the source's 3,673 APIs.
- **(b) Top-10 tool frequency** — the 10 most frequent tools and their share of examples, to quantify concentration.
- **(c) Train/validation match** — whether the 200-example validation split's tool distribution matches the training split's.

**Why it matters.**

- (a) quantifies coverage: if only a few hundred of 3,673 APIs appear, generalization to unseen tools is untested.
- (b) quantifies skew: if the top tool alone covers, say, >10% of examples, "tool-name accuracy" partly reflects memorization of frequent tools.
- (c) quantifies evaluation bias: a validation split whose tool mix differs from training would misstate the training objective.

**Method.** Ground-truth tool names are extracted from the first element of each example's `answers` list (the single expected call; the script also counts multi-answer rows). The validation split is compared with the training split using (i) the share of validation examples whose tool appears in training ("coverage"), (ii) Jensen–Shannon divergence over the pooled tool vocabulary, and (iii) a chi-square test of homogeneity on the top-10 tools with the remainder pooled into a single `<other>` category.

**Reference implementation** — `scripts/profile_tool_distribution.py` (reproduces `shuffle(seed=42) → select(5200) → 5000/200` using the `datasets` shuffle). Core logic:

```python
from datasets import load_dataset, Dataset

ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")  # gated: needs auth
subset = ds.shuffle(seed=42).select(range(5_200))
train, val = subset.select(range(5_000)), subset.select(range(5_000, 5_200))

def tool_name(ex):
    ans = ex["answers"][0] if isinstance(ex["answers"], list) else ex["answers"]
    return ans["name"]

from collections import Counter
train_c = Counter(tool_name(e) for e in train)
val_c   = Counter(tool_name(e) for e in val)
pooled  = train_c + val_c

# (a) unique tools                     (b) top-10
unique  = len(set(pooled))
top10   = pooled.most_common(10)

# (c) chi-square homogeneity on top-10 (+ "<other>") and Jensen-Shannon divergence
from scipy.stats import chi2_contingency
cats = [t for t, _ in top10] + ["<other>"]
rows = [[train_c.get(t, 0) for t in cats[:-1]] + [sum(c for t, c in train_c.items() if t not in dict(top10))],
        [  val_c.get(t, 0) for t in cats[:-1]] + [sum(c for t, c in   val_c.items() if t not in dict(top10))]]
chi2, p, dof, _ = chi2_contingency(rows, correction=False)
```

**Template results tables** (fill from the script's output):

(a) Unique tool count

| Quantity | Value |
| --- | --- |
| Unique ground-truth tool names in the 5,200-example subset | **TBD** |
| Unique APIs in full source (reference) | 3,673 |

(b) Top-10 tools

| Rank | Tool name | Count | Share of subset |
| --- | --- | ---: | ---: |
| 1 | `TBD` | TBD | TBD |
| 2 | `TBD` | TBD | TBD |
| 3 | `TBD` | TBD | TBD |
| 4 | `TBD` | TBD | TBD |
| 5 | `TBD` | TBD | TBD |
| 6 | `TBD` | TBD | TBD |
| 7 | `TBD` | TBD | TBD |
| 8 | `TBD` | TBD | TBD |
| 9 | `TBD` | TBD | TBD |
| 10 | `TBD` | TBD | TBD |
| remaining | *`<other>` tools* | TBD | TBD |

(c) Train vs. validation distribution match

| Check | Statistic | Value | Expected for a clean random split |
| --- | --- | ---: | --- |
| Validation examples covered by a tool seen in training | coverage | **TBD** | ≈ 100% |
| Jensen–Shannon divergence (train vs. val) | JSD | **TBD** | ≪ 0.05 |
| Chi-square homogeneity, top-10 + `<other>` | χ², p | **TBD** | p ≥ 0.05 |

**Why these values are TBD.** The source dataset is gated, and the exact `seed=42` sampling must run in the authenticated environment that produced the training data (the shuffle RNG is `datasets`-version-sensitive; the script prints the version it uses). The numbers must not be estimated or fabricated. Running `scripts/profile_tool_distribution.py` with a valid token produces the filled tables directly.

**How to read the results.**

- If the top tool's share is large (e.g., >10%), tool-name accuracy should be reported alongside the concentration, and the result should be caveated as partly reflecting high-frequency tools.
- If coverage < 100%, some validation tools are unseen in training — expected only if the split is not a clean random draw.
- p < 0.05 on the chi-square test indicates the validation distribution differs from training, which would bias the evaluation.

This item is tracked as **open** in §20 (caveats), §27.8 (limitations), and §39 (publication checklist).

# 9. Dataset Processing Methodology

Five stages:

```text
Source dataset → deterministic shuffle → sampling / split → ChatML formatting → tokenization → SFT dataset
```

1. **Shuffle** — `seed = 42`.
2. **Sampling** — 5,200 examples: 5,000 training, 200 validation.
3. **Tool serialization** — tool schemas are serialized into the prompt.
4. **ChatML formatting** — each example becomes *system → user (+ tools) → assistant (ground truth)*. The system instruction sets the structured-output constraint; the user message contains `Available Tools: <JSON schemas>` and `User Request: <query>`; the assistant target is the ground-truth JSON call.
5. **Tokenization** — the model's `chat_template` produces the training representation, keeping the training format aligned with the intended inference format.

# 10. Base Model

`Qwen/Qwen2.5-1.5B-Instruct` — a 1.5B-parameter instruction-tuned model, Apache-2.0 licensed. A small model is chosen deliberately: the objective is not maximum general capability but whether a small model can become **highly reliable on one structured-output task**.

# 11. Why QLoRA?

Full fine-tuning updates all parameters and demands substantial GPU memory. QLoRA combines a **4-bit quantized base model (NF4) + frozen base parameters + trainable low-rank adapters**, adding double quantization and paged optimizers to further reduce memory. LoRA freezes the original weights and learns low-rank update matrices, cutting trainable parameters and memory; QLoRA extends this to a quantized base. For TinyToolCaller this is a practical route to specializing a 1.5B model without full-model optimization.

# 12. Fine-Tuning Architecture

```text
Qwen2.5-1.5B-Instruct → 4-bit NF4 model → frozen base weights
                                              +
                                        LoRA adapters (trainable)
                                              ↓
                                          SFTTrainer
                                              ↓
                                        TinyToolCaller
```

LoRA configuration:

| Parameter | Value |
| --- | --- |
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Task type | CAUSAL_LM |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

# 13. Training Configuration

| Parameter | Configuration |
| --- | --- |
| Base model | Qwen2.5-1.5B-Instruct |
| Training / validation examples | 5,000 / 200 |
| Quantization | 4-bit NF4 |
| Double quantization | Enabled |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | 2e-4 |
| Scheduler / warmup | Cosine / 3% |
| Batch size / gradient accumulation | 2 / 8 |
| Effective batch size | 16 |
| Epochs | 2 |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 1024 |
| Precision | BF16 if available, otherwise FP16 |

Supervised fine-tuning uses Hugging Face TRL's `SFTTrainer`.

# 14. Clear Prerequisites and Requirements

**Hardware.** A CUDA-capable GPU. The QLoRA recipe is designed to fit single-GPU workflows, but exact requirements depend on GPU VRAM, CUDA/PyTorch/Transformers/bitsandbytes versions, batch configuration, and sequence length — verify against the environment before training.

**Software.** Python 3.x, PyTorch, CUDA, Transformers, Datasets, PEFT, TRL, bitsandbytes, Weights & Biases, Hugging Face Hub.

**Accounts.** A Hugging Face token (with gated access to the source dataset) and a W&B API key. The project uses `HF_TOKEN` and `WANDB_API_KEY` environment variables.

# 15. Implementation Considerations

Notable decisions, stated explicitly:

- **Markdown-wrapped output.** The baseline frequently wraps JSON in `` ```json `` fences. The evaluation strips these before parsing, so the 78.5% baseline JSON-validity figure **already benefits from cleanup** and is not raw-output purity (see §27.7).
- **Sequence length (1024).** Examples whose tool-schema lists exceed this are truncated. The pipeline does not report how many training examples were truncated or whether truncation biases the model toward smaller tool sets.
- **No retry/repair.** A single generation is scored as-is; production systems would retry malformed output before failing, which this evaluation does not simulate.

The pipeline itself is linear and inspectable: load tokenizer → load dataset → shuffle/split → ChatML formatting → baseline evaluation → load 4-bit model → prepare k-bit training → attach LoRA → train → save adapter → evaluate fine-tuned model → evaluate GSM8K → merge adapter → publish. For production software this should be modularized (`data/`, `model/`, `training/`, `evaluation/`, `inference/`, `publishing/`, `tests/`).

# 16. Code Usage Appropriateness

The library choices match the problem:

| Library | Purpose |
| --- | --- |
| Transformers | Model and tokenizer |
| Datasets | Dataset loading |
| PEFT | LoRA / parameter-efficient fine-tuning |
| TRL | Supervised fine-tuning |
| bitsandbytes | Quantization |
| PyTorch | Model execution |
| W&B | Experiment tracking |
| Hugging Face Hub | Artifact publication |

One configuration note: `paged_adamw_8bit` and gradient accumulation of 8 are standard for QLoRA on *larger* models under severe memory pressure. At 1.5B parameters with 4-bit quantization it is worth stating whether these were empirically necessary here or inherited from larger-model QLoRA tutorials without re-validation. If inherited without testing, say so explicitly — it is a legitimate simplification, but the current writeup otherwise reads as if every setting was tuned for this model.

# 17. Code Explanation Quality

The implementation is organized around explicit pipeline stages — *configuration, dataset preparation, evaluation functions, baseline evaluation, QLoRA fine-tuning, post-fine-tuning evaluation, merge & publish* — so the full model lifecycle can be traced from raw data to published model, with experiment parameters centralized in one configuration block. Future improvements: unit tests, type hints, modular files, configuration files, dataset validation, automated regression tests, configuration versioning, and more robust inference validation.

# 18. Evaluation Methodology

Three metrics are computed over the 200-example validation split:

| Metric | Definition | Computation |
| --- | --- | --- |
| JSON validity | Output contains a parseable JSON object | JSON extracted via regex/substring match, then `json.loads()`; **not** a raw-output purity check (§27.7) |
| Tool-name accuracy | Predicted `name` equals ground truth | Exact, case-sensitive string match |
| Argument exact match | Predicted `arguments` equals ground truth | Exact match on keys and values; no partial credit |

**Why exact match, not similarity scoring.** In a real execution pipeline, a partially correct argument set (right tool, wrong value) still fails downstream. Exact match reflects the deployment failure mode more honestly than a softer metric, at the cost of not distinguishing "close" from "way off" failures.

**Not measured.** Latency, token-level calibration, out-of-schema requests, and multi-tool selection when more than one tool could validly answer — candidates for a follow-up evaluation pass.

# 19. Baseline Results

| Metric | Result |
| --- | ---: |
| JSON validity | 78.5% |
| Tool-name accuracy | 65.0% |
| Argument exact match | 42.0% |
| GSM8K (n = 50) | 52.0% |

The base model often understands the request but adds markdown, explanatory text, selects the wrong tool, or omits arguments — establishing a meaningful baseline rather than evaluating the fine-tuned model in isolation.

# 20. Comparative Analysis

| Metric | Base | TinyToolCaller | Improvement |
| --- | ---: | ---: | ---: |
| JSON validity | 78.5% | **98.0%** | +19.5 pp |
| Tool-name accuracy | 65.0% | **92.5%** | +27.5 pp |
| Argument exact match | 42.0% | **84.0%** | +42.0 pp |

| Dimension | Base model | TinyToolCaller |
| --- | --- | --- |
| Argument exact match | 42.0% | 84.0% |
| Additional training cost | — | ≈5,000 examples, single GPU, QLoRA (hours, not days) |
| Additional inference cost | — | Negligible (LoRA adapter adds minimal latency/memory) |

> **Caveat — JSON validity.** Computed *after* JSON extraction from the raw output, not on the raw output itself. It measures whether a valid JSON object could be extracted, not whether the model produced pure JSON with no wrapper text. A stricter, deployment-relevant raw-output metric is not yet reported (§27.7).

> **Caveat — evaluation set.** All figures come from the same 200-example split used during development; no independent held-out test set was held back. These are **in-sample experimental results, not unbiased estimates of generalization** (§22).

> **Caveat — tool distribution (open item).** The subset's tool-name distribution has not yet been profiled (§8.1). Until (a) unique-tool count, (b) top-10 frequency, and (c) train/validation match are reported, the tool-name-accuracy figure cannot be separated from possible skew toward high-frequency tools.

The largest improvement is **argument exact match** (42.0% → 84.0%), the most deployment-relevant of the three. The direction is consistent across all three task-specific metrics, but the exact percentage-point improvements should not be read as precise estimates of unseen-data performance.

# 21. Catastrophic Forgetting Analysis

A subset of GSM8K probes whether specialization degraded a broader capability:

| Model | GSM8K (n = 50) |
| --- | ---: |
| Base | 52.0% |
| Fine-tuned | 50.0% |

A 2-point change on 50 examples is well within sampling noise: for a proportion near 50%, the approximate 95% confidence interval is **±14 percentage points**. This experiment therefore **cannot distinguish "no forgetting" from "moderate forgetting"** and should not be cited as evidence of retention either way. A stronger analysis would use a substantially larger set (ideally full GSM8K or another benchmark) under a fixed, identical harness for both models.

# 22. ⚠️ Evaluation Methodology Limitation

**Read before interpreting §20–§21.** The function-calling results reuse the 200-example split for development and final reporting — no independent test split was locked before experimentation. The GSM8K analysis uses 50 examples, insufficient for a strong retention claim. Therefore:

> The observed improvements are **directionally credible evidence** that fine-tuning improved behavior on the evaluated examples, but **not precise, unbiased estimates** of true accuracy on unseen data.

A stronger design separates the data into *training (5,000) → validation/dev (200) → independent test (never used for tuning)*, with the test set locked before experimentation and evaluated only after all training and selection decisions are finalized.

Despite these limits, three directional findings stand out on the evaluated examples:

1. **Structured output.** JSON extraction validity 78.5% → 98.0% — considerably more consistent structured output; note again that this is *after* extraction (§27.7), so 98% is **not** "98% of raw responses are pure JSON".
2. **Tool selection.** 65.0% → 92.5% — improved association of requests with available functions.
3. **Argument construction.** 42.0% → 84.0% — the largest gain, and the most practically important: selecting the right function is insufficient if the parameters are wrong.

The defensible conclusion: **QLoRA specialization substantially improved the model's observed ability to generate expected tool arguments on the evaluated examples** — while explicitly *not* establishing 84% argument exact match on an independently sampled real-world workload.

# 23. Deployment Considerations

TinyToolCaller is suitable for experimentation and model-level inference; production tool execution requires additional infrastructure.

| Control | Priority | Rationale (tied to measured failure rate) |
| --- | --- | --- |
| JSON Schema validation | Critical | 2% of fine-tuned outputs are not valid JSON — catch before execution |
| Tool allowlist | Critical | 7.5% tool-selection error means wrong-but-valid calls occur; allowlisting limits blast radius |
| Argument validation (type/range) | Critical | 16% argument mismatch — schema validation alone won't catch semantically wrong-but-well-typed values |
| Authorization / scoping | High | Independent of model accuracy — required regardless |
| Retry-with-repair | Medium | Could recover some of the 2% JSON-invalid cases cheaply |
| Audit logging | Medium | Post-hoc failure analysis |
| Rate limits / timeouts | Standard | Generic API hygiene |

A safe production architecture:

```text
User → Application → Tool Registry + Prompt Builder → TinyToolCaller → JSON output
      → JSON Schema Validator → Authorization → Tool Executor → External API
```

The model should not be granted unrestricted execution privileges.

# 24. Monitoring and Maintenance

Starting alert thresholds (calibrated against this experiment's baseline; revise after real traffic):

- **JSON validity < 95%** → page/alert (2 pp below the 98.0% eval figure).
- **Tool-selection failure rate > 10%** → investigate for drift (observed baseline: 7.5%).
- **Unknown/out-of-schema tool requests > 1%** → users are asking for capabilities outside the trained tool set.

Cadence: weekly failure-log review; a monthly regression run of the full evaluation suite against any new model version before promotion. The current project tracks training loss, learning rate, baseline/fine-tuned metrics, and system metrics (GPU utilization/memory) in W&B. A production system should extend this to **model metrics** (JSON validity, tool-selection and argument-validation failures, unknown-tool requests, retry rate, generation latency), **infrastructure metrics** (GPU memory, CPU/GPU utilization, throughput, request latency, error rate), and **data drift** (periodic comparison of production requests to the training distribution).

Maintenance loop:

```text
Production requests → failure analysis → evaluation dataset → regression test
    → fine-tuning → new model version → canary evaluation → deployment
```

# 25. Significance and Implications of the Work

The significance is **not** that a 1.5B model becomes universally more capable. The defensible finding is:

> **A small open-weight model can be substantially specialized for a narrowly defined structured-output task using parameter-efficient fine-tuning.**

This supports the engineering hypothesis that task specialization can sometimes be more valuable than increasing model size — relevant where inference cost, latency, hardware, or interface determinism matter. It also highlights a design principle: the **LLM interprets; deterministic software validates, authorizes, and executes** — a separation that makes AI systems easier to test and secure.

# 26. Industry Insights

As agentic systems increasingly interact with APIs, databases, search, calendars, and enterprise applications, the practical problem shifts from *"can the model generate a good answer?"* to *"can the model reliably produce an action software can safely execute?"* TinyToolCaller addresses one component of that transition, and explores the *large general-purpose vs. small specialized model* trade-off: a small specialized model can be attractive when the operational task is narrow enough that general reasoning is not the primary requirement. This project does **not** establish that smaller models are universally better — the right choice depends on task and schema complexity, latency, cost, error tolerance, deployment environment, and safety requirements.

# 27. Limitations Discussion

| Limitation | Likely impact on headline results |
| --- | --- |
| No independent test set (§27.1) | **High** — in-sample figures likely overstate generalization by an unknown margin |
| Small GSM8K sample (§27.2) | **High** — the retention claim is statistically unsupported either way |
| No hyperparameter sweep (§27.3) | **Medium** — gains may be improvable or a local optimum, not necessarily wrong |
| Limited training data, 5K/60K (§27.4) | **Medium** — more data would likely improve tool coverage; less clear effect on reported metrics |
| Single-turn focus (§27.5) | **Low for this report** — out of scope, doesn't bias current numbers |
| No external benchmark (§27.6) | **Medium** — limits comparability to other function-calling models |
| JSON-validity extraction leniency (§27.7) | **High** — 98.0% overstates raw-output compliance |
| Tool distribution unprofiled (§27.8) | **High** — until §8.1 is filled in, tool-name accuracy may be inflated by a skewed subset |

**27.1 No independent test set** — the 200-example validation split is also the final evaluation set; a future version should maintain train / validation / independent test.

**27.2 Small GSM8K sample** — 50 examples; the 2-point change is not a precise measurement of degradation.

**27.3 No hyperparameter sweep** — fixed configuration; no systematic search over learning rate, LoRA rank, dropout, epochs, sequence length, or batch configuration.

**27.4 Limited training data** — 5,000 examples of 60,000.

**27.5 Single-turn focus** — multi-turn tool use is not addressed.

**27.6 Limited benchmark coverage** — no results yet on a standardized external function-calling benchmark.

**27.7 Evaluation parsing** — JSON is extracted before scoring, so the JSON-validity metric does not represent the stricter requirement that the raw output contain *only* JSON. Correct this in a future evaluation version.

**27.8 Tool distribution unprofiled** — the subset's tool-name distribution (unique count, top-10 concentration, train/validation match) has not yet been measured (§8.1). Until those three quantities are reported, the tool-name-accuracy figure is entangled with possible selection skew, and the validation split's representativeness is unverified. This is the **highest-priority measurement to complete before publication**.

# 28. Source Credibility and Provenance

The project relies on first-party sources: the xLAM dataset is published by Salesforce AI Research with documented APIGen generation and verification; Qwen2.5-1.5B-Instruct is published by Qwen (Apache-2.0); QLoRA follows Dettmers et al. (4-bit NF4, double quantization, paged optimizers); PEFT/LoRA and TRL documentation come from Hugging Face; GSM8K originates from Cobbe et al.

# 29. Licensing and Attribution

| Artifact | License |
| --- | --- |
| Source dataset (`Salesforce/xlam-function-calling-60k`) | CC-BY-4.0 |
| Base model (`Qwen/Qwen2.5-1.5B-Instruct`) | Apache-2.0 |
| TinyToolCaller (code, derived data, documentation, artifacts) | Project-specific (see `LICENSE`) |

Users should review upstream licenses and attribution requirements before redistribution or commercial deployment.

# 30. Reproducibility

Core configuration:

```text
Seed 42 · 5,000 train / 200 validation · LoRA r=16, α=32, dropout=0.05
LR 2e-4 · cosine + 3% warmup · effective batch 16 · 2 epochs · max seq len 1024
```

Workflow:

```bash
pip install -r requirements.txt

export HF_TOKEN=<your_huggingface_token>       # must have gated access to the source dataset
export WANDB_API_KEY=<your_wandb_key>

python train_tool_caller.py
```

# 31. Repository and Dataset

| Artifact | Location |
| --- | --- |
| Code | https://github.com/strdst7/TinyToolCaller |
| Project dataset | https://huggingface.co/datasets/strdst77/TinyToolCaller |
| Source dataset | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k |
| Base model | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |

# 32. Project Architecture

```text
Salesforce xLAM 60K → sampling/split (5K train / 200 val) → ChatML formatting
    → Qwen2.5-1.5B-Instruct (4-bit NF4 + QLoRA) → SFTTrainer/TRL
    → LoRA adapter weights
        ├→ function-calling evaluation
        └→ GSM8K retention
    → model comparison → adapter + base → merged model → Hugging Face Hub
```

# 33. Future Architecture

A production-ready evolution adds a deterministic runtime around the model:

```text
User → Application → Tool Registry → Prompt Builder → TinyToolCaller → generated JSON
    → JSON Schema Validator → (invalid → repair/reject) → Authorization
    → Tool Executor → External API → tool result
```

The current project produces the **tool-call generation component**; a production agent adds validation, execution, permissions, retries, and observability.

# 34. Reader Next Steps

1. **Inspect the code** — start with the GitHub repository and the `CONFIG` block, then dataset preparation, evaluation, QLoRA, training, merge, publication.
2. **Inspect the datasets** — the derived project dataset, then the upstream xLAM 60K.
3. **Reproduce the baseline** — run base-model evaluation; record JSON validity, tool accuracy, argument exact match, GSM8K.
4. **Run the profiling** — run `scripts/profile_tool_distribution.py` and fill in §8.1 (unique count, top-10, train/val match).
5. **Run QLoRA** — use the documented configuration and training script.
6. **Compare** — base vs. fine-tuned under exactly the same evaluation procedure.
7. **Improve the experiment** — independent test set; larger GSM8K; multiple seeds; hyperparameter sweeps; external function-calling benchmarks; larger/more complex schemas; multi-turn conversations.
8. **Build the runtime** — tool registry + JSON Schema validation + authorization + execution + monitoring.

# 35. Recommended Research Extensions

Ordered by effort vs. impact:

**Low effort, high impact (do first):** independent test split; full GSM8K run (not 50 examples); **tool-distribution profiling of the training subset (§8.1)**.

**Medium effort:** hyperparameter sweep (rank, LR, epochs); external function-calling benchmark comparison.

**Larger research bets:** multi-turn tool use; distillation from a larger teacher model; runtime schema validation; 8-bit/4-bit inference optimization; standardized benchmarking.

# 36. Accessibility and Learning Design

The project is structured for readers with basic Python and ML knowledge but no prior LLM fine-tuning experience. The workflow is intentionally simple — *data → format → baseline → fine-tune → evaluate → compare → publish* — and readers need not understand every Transformer implementation detail. The concepts that matter: what function calling is; why structured JSON matters; what supervised fine-tuning does; what LoRA does; why quantization reduces memory; how baseline and post-training evaluation differ; and why validation methodology determines the strength of conclusions.

# 37. Key Takeaways

1. **Small models can be specialized** — a 1.5B model can become substantially better at a narrow structured-output task.
2. **QLoRA makes specialization accessible** — parameter-efficient fine-tuning reduces the trained state.
3. **Function calling is more than valid JSON** — tool selection and argument correctness are separate failure dimensions.
4. **Evaluation design matters** — evidence is only as strong as the evaluation methodology.
5. **LLMs should not be the entire agent** — combine the LLM with validation, authorization, and deterministic execution.

# 38. Conclusion

Starting from `Qwen/Qwen2.5-1.5B-Instruct` and fine-tuning with QLoRA on a 5,000-example subset of the Salesforce xLAM dataset, TinyToolCaller reports **JSON validity 78.5% → 98.0%, tool accuracy 65.0% → 92.5%, argument exact match 42.0% → 84.0%** on its 200-example evaluation split, with a 50-example GSM8K retention check moving 52.0% → 50.0%. The contribution is not a claim of universal superiority but a demonstrable engineering pattern:

```text
General-purpose model → task-specific data → parameter-efficient fine-tuning
    → specialized small model → structured interface → deterministic software
```

This pattern suits lower-cost, lower-latency AI systems whose target capability is narrow, measurable, and operationally well-defined. TinyToolCaller is best viewed as a **reproducible applied LLM engineering study and a foundation for a production-grade tool-calling runtime** — not a finished autonomous-agent platform.

# 39. Publication Checklist

- [x] Clear problem statement
- [x] Intended audience defined
- [x] Intended use case defined
- [x] Dataset source identified
- [x] Dataset provenance documented
- [x] Dataset processing explained
- [x] Training methodology documented
- [x] Prerequisites documented
- [x] Implementation workflow explained
- [x] Baseline established
- [x] Comparative results presented
- [x] Limitations disclosed
- [x] Deployment considerations discussed
- [x] Monitoring considerations discussed
- [x] Industry applications discussed
- [x] Source credibility documented
- [x] Licensing distinction corrected
- [x] Reproducibility instructions provided
- [x] Code repository linked
- [x] Dataset linked
- [x] Source dataset linked
- [x] Base model linked
- [x] Future work documented
- [x] Reader next steps provided
- [ ] **Tool-distribution profiling (§8.1) completed** — unique tool count, top-10 frequency, train/validation match *(open — blocks unqualified quoting of tool-accuracy)*
- [ ] Independent test set created and locked before model selection *(open)*
- [ ] Full-size GSM8K (or equivalent) retention run *(open)*

# 40. References

1. Salesforce AI Research. *xLAM Function Calling 60K / APIGen Function-Calling Datasets*. Hugging Face. https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
2. Qwen Team. *Qwen2.5-1.5B-Instruct*. Hugging Face. https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
3. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. *QLoRA: Efficient Finetuning of Quantized LLMs*. 2023. https://arxiv.org/abs/2305.14314
4. Hugging Face. *PEFT / LoRA Documentation*. https://huggingface.co/docs/peft/main/conceptual_guides/lora
5. Hugging Face. *TRL / SFTTrainer Documentation*. https://huggingface.co/docs/trl/sft_trainer
6. Cobbe, K. et al. *Training Verifiers to Solve Math Word Problems*. 2021. https://arxiv.org/abs/2110.14168

# 41. Project Links

- **Code** — https://github.com/strdst7/TinyToolCaller
- **Project dataset** — https://huggingface.co/datasets/strdst77/TinyToolCaller
- **Source dataset** — https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- **Base model** — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

---

## Final Project Statement

> **TinyToolCaller demonstrates that a small open-weight language model can be deliberately specialized for reliable function calling through QLoRA, achieving substantial improvements in structured output, tool selection, and argument accuracy while maintaining a reproducible and transparent evaluation pipeline. The project provides a practical foundation for exploring low-cost LLM agents in which the model generates structured intent and deterministic software remains responsible for validation, authorization, and execution.**

---

## Citing This Work

```bibtex
@misc{tinytoolcaller,
  title        = {TinyToolCaller: QLoRA Fine-Tuning of a 1.5B LLM for Reliable Function Calling},
  author       = {strdst7},
  year         = {2026},
  howpublished = {\url{https://github.com/strdst7/TinyToolCaller}},
  note         = {Base model: Qwen/Qwen2.5-1.5B-Instruct. Source dataset:
                  Salesforce/xlam-function-calling-60k.}
}
```
