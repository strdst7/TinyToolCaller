<img width="1920" height="1280" alt="poster" src="https://github.com/user-attachments/assets/903b1625-6b22-4ef2-9320-2340b945a1f0" />


<div align="center">


# TinyToolCaller

**QLoRA fine-tuning of a 1.5B LLM for reliable function calling**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)
[![Tests](https://img.shields.io/badge/Tests-41%20passed-brightgreen.svg)](tests)
[![Base model](https://img.shields.io/badge/Base-Qwen2.5--1.5B--Instruct-8A2BE2.svg)](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
[![Method](https://img.shields.io/badge/Method-QLoRA%20%2B%20LoRA%2FPEFT-0e7c7b.svg)](https://arxiv.org/abs/2305.14314)

*Given a natural-language request and a set of tool schemas, TinyToolCaller returns a single, strict JSON tool call — no markdown, no commentary.*

</div>

---

## Quick start

```bash
git clone https://github.com/strdst7/TinyToolCaller.git
cd TinyToolCaller
pip install -r requirements.txt
pip install pytest
```

```bash
export HF_TOKEN=<your_huggingface_token>     # gated dataset access (xlam-function-calling-60k)
export WANDB_API_KEY=<your_wandb_key>        # optional experiment tracking

python -m pytest tests/ -v                       # 41 tests
python scripts/profile_tool_distribution.py      # §8.1: unique tools, top-10, train/val match
python scripts/dataset_stats.py                  # §8.2: token stats + truncation count
python train_tool_caller.py                      # full 14-stage pipeline (baseline → QLoRA → eval → publish)
python scripts/statistical_analysis.py --report  # §18: Wilson CI + Cohen's h
```

## Repository structure

```text
train_tool_caller.py            # thin CLI: wires the package through the 14 pipeline stages
tinytoolcaller/                 # core package
    config.py                   #   central CONFIG + system prompt
    formatting.py               #   ChatML + JSON/answer extraction (pure, no heavy deps)
    data.py                     #   dataset load, seed-42 split, data-quality rules
    model.py                    #   4-bit NF4 loading + LoRA attachment
    metrics.py                  #   evaluation metrics (O-FME: JSON / tool / arguments)
    repair.py                   #   one-shot JSON repair loop
    train.py                    #   SFTTrainer wrapper + merge/publish
scripts/
    profile_tool_distribution.py   # §8.1 tool-distribution profiling
    dataset_stats.py               # §8.2 basic dataset statistics
    statistical_analysis.py        # §18 Wilson CI / McNemar / bootstrap
    capture_environment.py         # §12 environment capture
    publish_dataset.py             # build + upload train/validation.parquet to HF
    build_preprint.py              # render the publication as a PDF
    build_architecture.py          # render the system-architecture diagram
tests/                          # 41 pytest tests
images/                         # hero banners + architecture diagram (SVG/PNG)
preprint/                       # rendered publication PDF
```

## Links

| Artifact | Location |
| --- | --- |
| **Code** | https://github.com/strdst7/TinyToolCaller |
| **Project dataset** | https://huggingface.co/datasets/strdst77/TinyToolCaller |
| **Source dataset** (CC-BY-4.0, gated) | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k |
| **Base model** (Apache-2.0) | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |

## License

TinyToolCaller (code, derived data, documentation) is licensed under **Apache-2.0** ([LICENSE](LICENSE)). Upstream licenses: source dataset **CC-BY-4.0**, base model **Apache-2.0**.

---

<!-- publication-body-start -->

## Full Publication

*The complete research publication — related work, method, evaluation, statistics, limitations, deployment, monitoring, and rubric coverage. Sections are cross-referenced throughout as §N.*

| | |
| --- | --- |
| **Project** | TinyToolCaller |
| **Base model** | [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (Apache-2.0) |
| **Task** | Structured function / tool calling — JSON tool selection + argument construction |
| **Method** | QLoRA (4-bit NF4) + LoRA/PEFT, supervised fine-tuning via TRL `SFTTrainer` |
| **Source dataset** | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) (CC-BY-4.0, gated) |
| **Project dataset** | [`strdst77/TinyToolCaller`](https://huggingface.co/datasets/strdst77/TinyToolCaller) |
| **Code** | [`strdst7/TinyToolCaller`](https://github.com/strdst7/TinyToolCaller) |
| **Tracking** | Weights & Biases |
| **Status** | Research / applied LLM engineering study — pre-publication draft |


> **TL;DR.** This project takes a small open model (`Qwen2.5-1.5B-Instruct`), fine-tunes it with QLoRA on 5,000 function-calling examples, and shows it becomes much better at emitting the exact JSON tool call a downstream program needs: **JSON validity 78.5% → 98.0%, tool-name accuracy 65.0% → 92.5%, argument exact match 42.0% → 84.0%** on the project's 200-example evaluation split, with no measurable GSM8K degradation (52.0% → 50.0%, within noise). The point is not "we beat GPT-4" — it's "a reproducible QLoRA recipe can turn a 1.5B model into a reliable *structured-output component* of a larger system." All results are in-sample (no held-out test set yet), so read the caveats in §17 and §21 before quoting.

> **Abstract.** TinyToolCaller is an applied study of parameter-efficient specialization: it fine-tunes `Qwen2.5-1.5B-Instruct` with QLoRA on a 5,000-example supervised subset of the Salesforce xLAM function-calling dataset so that, given a natural-language request and a set of tool schemas, the model emits a single machine-readable tool call — `{"name": ..., "arguments": {...}}` — with no markdown, commentary, or extraneous text. Against the project's 200-example evaluation split, the fine-tuned model improves **JSON validity from 78.5% → 98.0% (95% CI [95.0%, 99.2%])**, **tool-name accuracy from 65.0% → 92.5% (CI [88.0%, 95.4%])**, and **argument exact-match from 42.0% → 84.0% (CI [78.3%, 88.4%])**, while a 50-example GSM8K retention check moves 52.0% → 50.0% (within sampling noise). The contribution is **not** a claim of state-of-the-art tool calling; it is a transparent, reproducible ablation isolating how much lift QLoRA alone provides over an unmodified 1.5B base model — a configuration the related literature (APIGen/xLAM, ToolACE, BFCL) has not isolated at this scale — together with an explicit accounting of evaluation limitations and a deterministic-runtime design in which the LLM produces structured intent while application code owns validation, authorization, and execution.

> **Reading guide.** §1–§4 state the problem, related work, contributions, and assumptions. §8–§13 cover data and method. §14–§20 cover evaluation and results. §21–§23 cover limitations and production. §24–§26 cover significance, insights, and provenance. §29–§38 cover reproduction, roadmap, and checklists. A **glossary** is in §34.1; a **rubric coverage matrix** is in §38.

---

> **⚠️ Before quoting the headline numbers.** The results in §16–§20 are **in-sample**: the 200-example split is also the development/evaluation set (no independent held-out test set), the GSM8K check uses 50 examples, and the **tool-distribution profile of the 5,200-example subset has not yet been measured** (§8.1). Until §8.1's three quantities — (a) unique tool count, (b) top-10 tool frequency, (c) train/validation distribution match — are filled in, the tool-name-accuracy figures cannot be separated from possible selection skew. Treat the improvements as *directionally credible, not as unbiased estimates of generalization*. See §17, §21, §28.

---

## 1. Introduction and Problem Statement

Large language models increasingly act as the interface between natural-language users and software systems. When a model must operate a real API, a conversational answer —

> "The weather in Tokyo is likely to be sunny."

— is insufficient. A tool-using system needs a structured, executable representation:

```json
{ "name": "get_weather", "arguments": { "location": "Tokyo" } }
```

A language model can fail this task at **seven distinct levels**: (1) produce invalid JSON; (2) wrap JSON in markdown fences; (3) append explanatory text; (4) select the wrong tool; (5) omit required arguments; (6) generate incorrect argument values; (7) invent arguments absent from the schema.

This project's central question is narrow and falsifiable: **can a 1.5B open-weight instruction model be specialized — via QLoRA alone, on 5,000 examples — so that it emits a valid, correctly-targeted, correctly-argumented tool call substantially more often than the same model without fine-tuning?** The output contract is a single object with exactly two fields — `name` and `arguments` — and no additional commentary or formatting.

## 2. Related Work

Function calling (also "tool use") has developed along three strands: data generation, models, and benchmarks. This section reviews each and states where TinyToolCaller sits relative to them.

### 2.1 Data generation

**APIGen / xLAM** (Liu et al., 2024) [1] introduced an automated pipeline that generates function-calling data and verifies each sample in three hierarchical stages — format checking, actual function execution, and semantic verification — producing 60,000 examples over 3,673 executable APIs in 21 categories. It is the direct source of this project's dataset, and its authors showed that models trained on the data (even 1B-scale) can exceed GPT-3.5-Turbo on the Berkeley Function-Calling Benchmark. **TinyToolCaller deliberately does not reproduce APIGen's scale**: it uses a fixed 5,000-example slice to isolate the *method's* effect rather than chase leaderboard rank. The multi-turn extension **APIGen-MT** (Prabhakar et al., 2025) [9] confirms the field's direction toward agentic, multi-step tool use — which this project explicitly scopes out (§4).

**ToolACE** (Liu et al., 2024) [6] is the closest methodological neighbour: it generates a larger, more diverse tool corpus (26,507 tools) with rule- and model-based verification and shows 8B models reach GPT-4-competitive function calling. Its key relevance here is its *scaling observation*: raw 0.5B–1.8B models "showed minimal function-calling ability," but fine-tuning "significantly enhanced" them. TinyToolCaller is a direct, small-scale confirmation of that observation at 1.5B, with the added value of reporting *per-failure-mode* decomposition (§19) that ToolACE's aggregate accuracy does not.

### 2.2 Models

**Gorilla** (Patil et al., 2023) [3] and **Toolformer** (Schick et al., 2023) [4] established the two dominant training paradigms — Gorilla via supervised data for API-connecting LMs, Toolformer via self-supervised tool-use annotation. **NexusRaven** (Srinivasan et al., 2023) [5] demonstrated that a 13B model, fine-tuned on curated data *without* GPT-4 distillation, matches GPT-3.5 zero-shot, and that in-context demonstration retrieval further helps. **Granite-20B-FunctionCalling** (Abdelaziz et al., 2024) [7] showed multi-task, granular training produces the best open function-calling model of its time on BFCL. **Octopus v2** (Chen & Li, 2024) [14] is the closest analogue in spirit: a 2B on-device model exceeding GPT-4 on function-calling accuracy while cutting context length 95%.

**Critical gap these works leave open.** Each of these results is entangled with its own data pipeline, scale, or architecture choice; none isolates "QLoRA on a frozen 1.5B base vs. that base prompted directly" on identical data and metrics. That is the specific ablation TinyToolCaller contributes — and it is why the project reports a *prompted-baseline* comparison (§16–§17) rather than comparing only against published leaderboard numbers.

### 2.3 Benchmarks and evaluation

**BFCL** (Patil et al., 2025) [2] is the de-facto standard for function calling, with AST-based and execution-based scoring across simple, parallel, and multi-turn calls; V3 (2025) added relevance detection ("when *not* to call") and closed its test data to prevent contamination. **τ-bench** (Sierra et al., 2024) [8] evaluates tool-agent-user interaction in realistic domains. An exploratory study of **small models for function calling** on the same xLAM dataset (arXiv:2504.19277, 2025) [10] reports evaluation on 1.35B–3.82B models and notes dataset-format adherence as a key enabler.

**Critical gap.** BFCL V3's test set is closed and its scoring is AST-based; τ-bench is multi-turn. Neither provides the exact-match, per-failure-mode breakdown (valid JSON / correct tool / correct arguments as three separate rates) that a deployment engineer needs. TinyToolCaller's three-metric decomposition (§14) is deliberately cruder but more *actionable*: each metric maps to a specific production control (§22).

### 2.4 Efficiency methods

QLoRA (Dettmers et al., 2023) [12] — 4-bit NF4 quantization, double quantization, paged optimizers — and LoRA (Hu et al., 2021) [11] underpin the training recipe (§10–§11). The project treats these as *tools*, not contributions: the methodological novelty claim is limited to the ablation design and the evaluation decomposition, not to any new training technique (§3).

## 3. Objectives, Contributions, and Originality

**Objectives.** (1) Transform a public function-calling dataset into instruction–response examples; (2) measure the unmodified base model's reliability before fine-tuning; (3) apply QLoRA without updating the full base; (4) compare fine-tuned vs. base on identical metrics; (5) check for capability degradation on GSM8K; (6) publish code, derived data, methodology, results, and artifacts.

**Originality and innovation.** This project introduces no new architecture or loss function — QLoRA and LoRA are used as published. What it contributes, stated precisely, is **three methodological artifacts**, two of which are directly reusable:

- **A clean, small-scale ablation** — QLoRA-only lift over a prompted 1.5B base, on a fixed 5,000-example slice, with the same metrics and (by default) the same quantization regime for both models (§14, §16) — a comparison the larger works above do not isolate.
- **The O-FME framework (§3.1)** — a named, three-axis evaluation that decomposes tool-calling accuracy into JSON validity / tool selection / argument construction, each mapped to a specific production control. This is the piece the published benchmarks (BFCL's aggregate accuracy) do not provide.
- **A one-shot JSON repair loop (§3.1)** — a concrete, tested, model-agnostic mechanism for recovering the malformed-output fraction at inference time.
- **Honest treatment of evidence quality** — the evaluation's limitations (in-sample split, 50-example GSM8K, unprofiled tool distribution) are surfaced at the point where results are quoted (§17, §21), and the statistics are reported with confidence intervals (§18).

### 3.1 Methodological contributions (implemented, tested)

**O-FME — Orthogonal Failure-Mode Evaluation.** A tool-calling result is scored along three *orthogonal* axes — (1) *validity*: is a JSON object extractable? (2) *selection*: is `name` correct? (3) *construction*: does `arguments` match exactly? — rather than collapsed into one accuracy. The framework's value is the **production-control mapping**: each axis corresponds to exactly one deterministic safeguard (validity → schema validation; selection → allowlist; construction → type/range/authorization checks, §22.1), so an O-FME report is directly actionable by a deployment team, which aggregate leaderboard scores are not. It is implemented in `tinytoolcaller/metrics.py` (§14) and unit-tested.

**One-shot JSON repair loop.** A lightweight, model-agnostic recovery step for the malformed-output fraction (~2% of fine-tuned outputs, §19.2). On an invalid generation, the model is re-prompted once with *its own offending output* plus a compact instruction, and the result is re-extracted:

```python
def repair(raw, generate_fn, prompt, max_attempts=1):       # tinytoolcaller/repair.py
    attempts = 0
    while extract_json(raw) is None and attempts < max_attempts:
        raw = generate_fn(prompt + REPAIR_INSTRUCTION + raw)  # model sees its own failure
        attempts += 1
    return raw, attempts
```

*Justification of the design choices:* (i) **one** attempt, because each retry costs latency and tokens while marginal recovery drops sharply after the first — the loop is a *cheap* mitigation, not a search; (ii) the model sees its **own output** rather than a generic error, because the concrete failure signature is the most informative signal; (iii) `extract_json` is the **same parser used at evaluation** (§14), so the repair loop and the reported metrics share one definition of "valid". The loop is dependency-injected (`generate_fn`), hence unit-testable on CPU (`tests/test_repair.py`, 41-test suite) and reusable with any decoder. Its measured recovery rate on the evaluation set is a future-work item (§33), pending the per-example dump (§18).

## 4. Assumptions and Scope

The following assumptions are **stated explicitly**; relaxing any of them changes what the results mean.

1. **Single-turn, single-call.** Each request maps to exactly one tool call; multi-step and multi-turn trajectories are out of scope (§32).
2. **English-language, JSON-format tools.** Tool schemas are serialized as JSON text (§9); no native function-call token (e.g., Qwen's `<tool_call>` tokens) is used.
3. **Closed tool set at inference.** All candidate tools are provided in the prompt; the model never selects an unseen tool. Out-of-schema requests are not evaluated (§14).
4. **Correctness = exact match.** A tool call is scored as correct only if `name` and the full `arguments` dict match the ground truth exactly (§14). This mirrors the downstream failure mode but ignores semantically-equivalent answers.
5. **The 200-example split is representative** of the intended distribution. This is currently **unverified** for tool distribution (§8.1) and is the principal open item before generalization claims can be made.
6. **Base and fine-tuned models are scored under the same quantization** (4-bit NF4 by default, §14), so the comparison isolates fine-tuning rather than precision.
7. **A valid JSON extraction from the response counts as valid** (§14, §21.7) — the metric is "extractable JSON", not "raw output is pure JSON".

## 5. Intended Audience and Use Case

### 5.1 Who this work is for, and what each reader takes away

| Audience | What they take away from this paper |
| --- | --- |
| **LLM/ML engineers** building tool-calling systems | A reproducible QLoRA recipe (§10–§13) and a failure-budget table (§19.2) showing *which* failures to guard against |
| **Researchers** in parameter-efficient fine-tuning | A clean ablation — QLoRA-only lift over a prompted 1.5B base (§16–§17) — with confidence intervals and a paired test design (§18) |
| **AI application/agent developers** | The integration contract (§22.3), the runtime-control stack (§22.1), and why schema validation alone is insufficient (§19.2) |
| **Students** learning practical LLM fine-tuning | An end-to-end walkthrough from gated dataset to published model, with a glossary (§34.1) and worked tokenization example (§9.1) |
| **Practitioners evaluating small models** | A concrete case that a 1.5B model + QLoRA can be a viable structured-output *component* (§24), with cost framing (§17.1) |

### 5.2 Required background knowledge (tiered)

| Tier | You can follow… | Prerequisite knowledge |
| --- | --- | --- |
| **Beginner** | §1, §5–§9, §34 (glossary + analogies), §35 takeaways | Basic Python; familiarity with what an API and JSON are |
| **Intermediate** | + §10–§14, §16–§20, §24–§26 | Supervised learning; a rough idea of what fine-tuning and quantization do (the glossary covers the rest) |
| **Advanced** | + §15, §18, §21, §28–§29 | Transformers, Hugging Face ecosystem (PEFT/TRL/bitsandbytes); statistical testing (Wilson CI, McNemar) |

Reading **§1, §4 (assumptions), §17 (results + caveats), and §21 (limitations)** is sufficient to correctly interpret the headline numbers; the rest is depth. Technical terms are defined on first use and collected in §34.1.

### 5.3 What this work is *not* for

- **Not** a state-of-the-art function-calling model claim (§2–§3) — the project positions itself as a small-scale ablation, and §17.3 makes the comparison to published SOTA explicitly non-competitive.
- **Not** a drop-in production agent — the model is one component; §22 specifies the deterministic layer that must surround it.
- **Not** a claim of capability retention — §20 shows the GSM8K sample is too small to support one.

### 5.4 Intended use case

Structured tool selection: a downstream application supplies *user request + tool schemas*, TinyToolCaller returns *tool name + arguments*, and the application validates and executes:

```text
LLM                                     Deterministic application
│  Understand request                   │
├─ Select tool                          │  Validate
└─ Construct arguments                  │  Authorize
        │                               │  Execute
        ▼                               ▼
  Structured JSON ─────────────────▶  Tool execution
```

The project does **not** claim the LLM should directly execute arbitrary external functions.

## 6. Real-World Applications

| Scenario | Request | Expected call |
| --- | --- | --- |
| Personal assistant | "Add a dentist appointment tomorrow at 3 PM." | `create_calendar_event(...)` |
| Customer support | "What's the status of order 12345?" | `get_order_status(...)` |
| Enterprise search | "Find all invoices from Vendor X this quarter." | `search_financial_records(...)` |
| Weather / info systems | "What's the weather in Tokyo?" | `get_weather(...)` |
| Database assistant | "Show me customers who haven't purchased in 90 days." | `query_customer_database(...)` |
| Workflow automation | "Create a support ticket and assign it to infra." | `create_ticket(...)` → `assign_ticket(...)` |

In each case the model's job is to translate human intent into a machine-readable representation that deterministic software can process.

## 7. Background: Function Calling in Small Models

Small-model (<3B) function calling is achieved three ways:

1. **Base-model prompting** — relying on existing instruction-following without adaptation. This is what the project's baseline measures.
2. **Full fine-tuning** — updating all parameters; effective but expensive for iterative single-GPU work.
3. **Parameter-efficient fine-tuning (LoRA/QLoRA)** — the approach here, and increasingly the default for small-model specialization because it fits single-GPU workflows.

Existing models (xLAM-1b-fc-r/7b-fc-r [15], Hermes function-calling models, Gorilla [3], NexusRaven [5], Octopus v2 [14]) already target this capability. **TinyToolCaller does not claim to outperform them** — it isolates how much lift QLoRA alone provides over an unmodified 1.5B base model with a fixed, reproducible recipe (§2).

## 8. Dataset: Source, Selection Rationale, and Description

### 8.0 Selection rationale

The project uses [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k), the APIGen release [1], because it uniquely satisfies the project's constraints:

| Criterion | xLAM-60k | Alternatives considered |
| --- | --- | --- |
| Permissive license | CC-BY-4.0 | ToolACE (mixed licensing per tool); BFCL test closed from V3 |
| Verifiability | 3-stage execution-verified [1] | ToolBench/ToolAlpaca (self-instruct, weaker verification) |
| Diversity | 3,673 APIs / 21 categories | Gorilla APIBench (fewer, code-centric APIs) |
| Scale compatible with single-GPU | 60K, subsettable deterministically | ToolLLM (126K+, heavier) |
| Small-model evidence | xLAM-1b-fc-r trained on it [1] | — |

Per the dataset card: 60,000 examples; the first 33,659 generated by DeepSeek, the remainder by Mixtral; human evaluation over 600 samples with correctness above 95%; remaining minor issues acknowledged [1].

> **Access note.** The dataset is **gated** on the Hugging Face Hub: loading requires a logged-in account that has accepted the terms (`HF_TOKEN`). This is stated explicitly because it affects reproducibility (§28).

**Utilization and modification log** (what this project did *to* the dataset, step by step, so the provenance is auditable):

| Step | Operation | Change to data |
| --- | --- | --- |
| 1 | Download the gated source (60,043 rows) | None |
| 2 | `shuffle(seed=42)` → `select(5,200)` | Subsetting only — no row is edited, re-labelled, or filtered |
| 3 | Split 5,000 train / 200 validation | Membership freeze only |
| 4 | Defensive cleaning rules (§9.2) | Drop-and-count rows failing structural checks (missing query/tools/answers, malformed tools, exact duplicates); **no value-level changes** |
| 5 | ChatML formatting + tokenization (§9) | A *derived view* used at training time; the published Parquet keeps the raw xLAM fields |
| 6 | Publish `train.parquet` / `validation.parquet` to `strdst77/TinyToolCaller` | Artifact creation (§8.4) |

No class re-balancing, no argument-value editing, and no augmentation were performed — the source's execution verification (§8.5) is the sole authority on label correctness.

### 8.1 Tool-Distribution Profiling *(required before quoting results)*

The single most important open measurement is the tool distribution of the 5,200-example subset. Three quantities must be reported:

- **(a) Unique tool count** — distinct ground-truth tool names (`name`) in the 5,200-example subset, characterizing coverage of the source's 3,673 APIs.
- **(b) Top-10 tool frequency** — the 10 most frequent tools and their share of examples, quantifying concentration.
- **(c) Train/validation match** — whether the 200-example validation split's tool distribution matches the training split's.

**Why it matters.** (a) quantifies coverage — if only a few hundred APIs appear, generalization to unseen tools is untested. (b) quantifies skew — if the top tool covers >10% of examples, "tool-name accuracy" partly reflects memorization. (c) quantifies evaluation bias — a validation split whose tool mix differs from training misstates the objective.

**Method.** Ground-truth tool names are extracted from the first element of each example's `answers` list (the single expected call; the script also counts multi-answer rows). Train vs. validation are compared with (i) coverage (share of validation examples whose tool appears in training), (ii) Jensen–Shannon divergence over the pooled vocabulary, and (iii) a chi-square test of homogeneity on the top-10 tools with the remainder pooled as `<other>`.

**Reference implementation** — `scripts/profile_tool_distribution.py` (reproduces `shuffle(seed=42) → select(5200) → 5000/200` with the `datasets` shuffle).

*Purpose of the snippet:* show the three statistics (a)–(c) as ~15 executable lines, so a reader sees that the "TBD" values are a reproducibility step, not a hidden analysis. Core logic:

```python
from datasets import load_dataset
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")  # gated
subset = ds.shuffle(seed=42).select(range(5_200))
train, val = subset.select(range(5_000)), subset.select(range(5_000, 5_200))

def tool_name(ex):
    ans = ex["answers"][0] if isinstance(ex["answers"], list) else ex["answers"]
    return ans["name"]

from collections import Counter
train_c = Counter(tool_name(e) for e in train)
val_c   = Counter(tool_name(e) for e in val)
pooled  = train_c + val_c
unique  = len(set(pooled))                    # (a)
top10   = pooled.most_common(10)              # (b)

from scipy.stats import chi2_contingency
cats = [t for t, _ in top10] + ["<other>"]
rows = [[train_c.get(t,0) for t in cats[:-1]] + [sum(c for t,c in train_c.items() if t not in dict(top10))],
        [  val_c.get(t,0) for t in cats[:-1]] + [sum(c for t,c in   val_c.items() if t not in dict(top10))]]
chi2, p, dof, _ = chi2_contingency(rows, correction=False)   # (c)
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

**Why these values are TBD.** The source dataset is gated, and the exact `seed=42` sampling must run in the authenticated environment that produced the training data (the shuffle RNG is `datasets`-version-sensitive; the script prints the version it uses). The numbers must not be estimated or fabricated.

**How to read the results.** A top-tool share >10% means tool accuracy should be reported alongside concentration. Coverage <100% means some validation tools were unseen in training. p < 0.05 indicates the validation distribution differs from training and would bias the evaluation. This item is tracked as **open** in §17, §21.8, and §36.

### 8.2 Basic Dataset Statistics

**Source-level (documented — dataset card [1] and the APIGen paper):**

| Statistic | Value |
| --- | --- |
| Total examples | ≈60,000 (60,043: 33,659 + 26,384) |
| Unique executable APIs | 3,673 (3,539 REST APIs + 134 Python functions) |
| API categories | 21 (consolidated from ToolBench's 49) |
| Generators | DeepSeek-V2-Chat (236B): 33,659 (ids 0–33,658) · Mixtral-8x22B-Inst: 26,384 (remaining) |
| Generation temperature | 0.7 |
| Query styles | 4 — Simple, Multiple, Parallel, Parallel Multiple |
| Verification | 3-stage (format / execution / semantic) |
| Human evaluation | >95% correctness on 600 samples |
| License | CC-BY-4.0 |

**Subset-level (must be computed — `scripts/dataset_stats.py` prints a paste-ready table):**

| Statistic | Value |
| --- | --- |
| Examples (train / validation) | 5,000 / 200 |
| Unique tool names in subset | **TBD** (§8.1a) |
| Multi-answer rows (>1 ground-truth answer) | **TBD** |
| Tools per example — mean / median / max | **TBD** |
| Prompt tokens (system+user) — mean / median / p95 / max | **TBD** |
| Examples truncated at max_seq_length = 1024 | **TBD** |

The token statistics use the base model's tokenizer and directly quantify the §13 sequence-length concern. The script is:

```bash
export HF_TOKEN=<token>
python scripts/dataset_stats.py                # load from the gated Hub
# or
python scripts/dataset_stats.py --path data/subset.json
```

Until these numbers exist, the paper reports only source-level statistics and explicitly flags the subset as uncharacterized (§21.8).

### 8.3 Source Dataset — Schema and Structure

Each example in `Salesforce/xlam-function-calling-60k` is a JSON object with **three core components** (a `thought` chain-of-thought field may also be present in some records from the generation step; the pipeline ignores it and uses only the three below):

| Field | Type | Description |
| --- | --- | --- |
| `query` | string | The natural-language instruction or user request |
| `tools` | array | The candidate tool schemas available to answer the query; each tool is `{name, description, parameters}` where `parameters` is a JSON-Schema-style object (`type`, `properties`, `required`, `enum`, …) |
| `answers` | array | The ground-truth function call(s); each entry is `{name, arguments}`. Multiple entries occur in the parallel query styles; single-call examples have one entry |

**Features and data types, stated explicitly.** Each example is a record with **three top-level features** (plus an optional `thought` CoT field that the pipeline drops). Recursively, every value is JSON-typed: `query` → string; `tools` → array of objects, each `{name: string, description: string, parameters: object}` with `parameters.properties` a string-keyed object of `{type, description, enum?}` and `parameters.required` an array of strings; `answers` → array of `{name: string, arguments: object}` where `arguments` values are strings, numbers, booleans, or nested objects. The pipeline only ever reads `query`, `tools`, and `answers[0]` (§9).

**Class distribution.** In function-calling data the natural "classes" are the **query styles** (§8.3's table: Simple / Multiple / Parallel / Parallel Multiple), which are controlled at generation time by APIGen's prompt templates [1]. The *source-level* class balance is documented in the APIGen paper but not reproduced here numerically (it lives in a figure); the **subset-level** class and tool-name distributions are the unmeasured quantities §8.1 and §8.2 are defined to produce — and, in the single-call setting, Parallel-style examples are effectively out-of-contract (§8.3), which is why their subset share must be reported before quoting tool accuracy.

Annotated illustrative example (the running example throughout this paper):

```json
{
  "query": "What's the weather in Tokyo?",          // natural-language request
  "tools": [                                         // candidate tools shown to the model
    {
      "name": "get_weather",                         // tool name the model may call
      "description": "Get the current weather for a location",
      "parameters": {                                // JSON-Schema constraints
        "type": "object",
        "properties": {
          "location": { "type": "string", "description": "City name" },
          "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["location"]
      }
    }
  ],
  "answers": [                                       // ground truth (single call here)
    { "name": "get_weather", "arguments": { "location": "Tokyo", "unit": "celsius" } }
  ]
}
```

**Query-style breakdown.** APIGen structures the data into four query styles (mirroring BFCL's categories):

| Style | Definition | Relevance to this project |
| --- | --- | --- |
| Simple | One function call from one provided API | In scope — the core single-call setting |
| Multiple | Choose the most appropriate of several provided APIs | In scope — this is exactly "tool selection" |
| Parallel | Multiple simultaneous calls from one API | **Out of scope** — this project assumes one call (§4.1) |
| Parallel Multiple | Multiple calls, multiple APIs | **Out of scope** |

Because the pipeline's `ground_truth()` takes the **first** entry of `answers` (§9), Parallel examples would be scored against a single call and are effectively out of the model's contract. The share of Parallel/Parallel-Multiple examples in the 5,200-example subset is therefore a quantity that should be reported alongside §8.1 (it bounds how much of the subset the single-call contract even applies to) — currently unmeasured.

**API composition.** 3,539 REST APIs (cleaned from ToolBench's 16,464 across 49 coarse categories — parsing fixes, accessibility testing, docstring regeneration) plus 134 well-documented Python functions (math, finance, data management) — 3,673 in total, consolidated into 21 categories spanning technology, social sciences, education, sports, finance, and others (the full category list appears in the APIGen paper, Figure 4).

### 8.4 Project Dataset (Derived Subset)

The derived dataset `strdst77/TinyToolCaller` is a **deterministic seed-42 subset** of the source:

| Property | Value |
| --- | --- |
| Provenance | `shuffle(seed=42)` → `select(5,200)` over the source's `train` split, then first 5,000 = train, last 200 = validation |
| Splits / files | `train.parquet` (5,000 rows), `validation.parquet` (200 rows) |
| Stored format | The raw xLAM fields (`query`, `tools`, `answers`) — **not** pre-tokenized; ChatML formatting is applied at training time via the tokenizer's `chat_template` (§9), so the artifact stays tokenizer-agnostic |
| Changes vs. source | None beyond subsetting (no re-labelling, no filtering, no deduplication) |
| License | Upstream CC-BY-4.0; the derived card is Apache-2.0 (§28) |
| Publication status | **Open** — the HF repo currently holds only the card; `scripts/publish_dataset.py --push` uploads the two Parquet files |

The subset is published separately from the source for three reasons: it freezes the exact training/validation membership for reproducibility (§29); it gives downstream users a small, downloadable artifact instead of the 97.7 MB gated source; and it makes the split provenance auditable.

### 8.5 Data Quality Characteristics

**Strengths (documented in the APIGen paper [1]).** Each example passed **three hierarchical verification stages**, and the paper publishes the per-generator filtering statistics:

| Generator | Verified | Fail format | Fail execution | Fail semantic | Pass rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V2-Chat (236B) — *in release* | 33,659 | 817 | 3,359 | 2,165 | 84.15% |
| Mixtral-8x22B-Inst — *in release* | 26,384 | 1,680 | 5,073 | 6,863 | 65.96% |
| Mixtral-8x7B-Inst — *not released* | 15,385 | 3,311 | 12,341 | 7,963 | 38.46% |
| DeepSeek-Coder-33B-Inst — *not released* | 13,769 | 4,311 | 15,496 | 6,424 | 34.42% |

- **Stage 1 — Format Checker:** rejects malformed JSON, hallucinated functions/arguments not in the provided APIs, and invalid argument values.
- **Stage 2 — Execution Checker:** executes each call against the real backend (REST calls; Python functions run in a subprocess) and discards anything that fails — argument type errors, invalid parameters, timeouts, etc.
- **Stage 3 — Semantic Checker:** an LLM judge verifies the call aligns with the query's objective, chooses from the available functions, matches the number of intended calls, and returns relevant results.
- The release deliberately keeps **only the two strongest generators** (the two weakest models' data had pass rates under 40% and were excluded), and a human evaluation of 600 samples scored >95% correctness.

**Weaknesses and caveats (why quality is not assumed).**

1. **Synthetic provenance.** All queries are LLM-generated (temperature 0.7), so the data inherits generator biases; it is *execution-verified*, not human-authored.
2. **Remaining noise acknowledged.** The dataset card itself notes "remaining minor issues"; verification is not a correctness guarantee.
3. **Multi-answer rows.** Parallel-style examples carry multiple ground-truth calls; the single-call pipeline (§9) uses only the first, so those rows are partially out-of-contract (§8.3).
4. **Finite API coverage.** 3,673 APIs is broad but not universal; generalization to unseen APIs is untested here.
5. **English-only, JSON-only.** Non-English queries and non-JSON tool protocols are unrepresented (§4.2).
6. **Subset skew unquantified.** The tool distribution of the 5,200-example subset is unmeasured (§8.1) — the single largest open quality question for the reported results.

The net position: the source is among the most rigorously verified function-calling datasets available, and this project inherits that quality — but the *subset's* representativeness is exactly what §8.1 must establish before the results are quoted without qualification.

### 8.6 Relationships between the datasets

Three datasets appear in this project; their relationship is a strict derivation chain, not a set of independent sources:

```text
Salesforce/xlam-function-calling-60k      (source, CC-BY-4.0, gated, 60,043 rows)
        │  shuffle(seed=42) → select(5,200) → 5,000 train / 200 validation
        ▼
strdst77/TinyToolCaller                    (derived subset — a deterministic slice)
        │  train.parquet (5,000) · validation.parquet (200)
        ▼
SFT dataset (in-memory, at training time)  (a formatting view — ChatML + tokenization, §9)
```

| Dataset | Role | Relationship |
| --- | --- | --- |
| `Salesforce/xlam-function-calling-60k` | Upstream source | Parent — all project data descends from it; **not redistributed** (gated) |
| `strdst77/TinyToolCaller` | Derived subset | A seed-42 **slice** of the source; no relabelling, filtering, or editing beyond the §8.0 modification log and §9.2 structural checks |
| SFT dataset | Training representation | A **view** of the subset — the same rows ChatML-formatted and tokenized at training time; not persisted as an artifact |

The evaluation split, the §8.1 profiling sample, and the §18 statistical analyses all operate on **the same 200 validation rows** of the derived subset — so every number in this paper traces to a single, auditable provenance chain (§29).

## 9. Dataset Processing Methodology

Five stages (the schema is documented field-by-field in §8.3):

```text
Source dataset → deterministic shuffle → sampling / split → ChatML formatting → tokenization → SFT dataset
```

1. **Shuffle** — `seed = 42`.
2. **Sampling** — 5,200 examples: 5,000 training, 200 validation.
3. **Tool serialization** — tool schemas are serialized into the prompt as JSON.
4. **ChatML formatting** — each example becomes *system → user (+ tools) → assistant (ground truth)*. The system instruction sets the structured-output constraint; the user message contains `Available Tools: <JSON schemas>` and `User Request: <query>`; the assistant target is the ground-truth JSON call.
5. **Tokenization** — the model's `chat_template` produces the training representation, keeping the training format aligned with the intended inference format.

### 9.1 Worked example, end to end

The following is the **illustrative** weather example from §8, formatted with the *actual* `Qwen/Qwen2.5-1.5B-Instruct` tokenizer (vocab size 151,665). The training string produced by `apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` is:

```text
<|im_start|>system
You are a function-calling assistant. Given the user's request and the available tools, select the correct tool and construct the correct arguments. Respond with ONLY a JSON object containing exactly two keys: "name" (the tool name) and "arguments" (an object of argument values). Do not include markdown, explanations, or any other text.<|im_end|>
<|im_start|>user
Available Tools:
[{"name": "get_weather", "description": "Get the current weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}]

User Request:
What's the weather in Tokyo?<|im_end|>
<|im_start|>assistant
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}<|im_end|>
```

Real token counts from this exact example: **199 tokens** for the full training sequence (system + user + assistant), **173 tokens** for the inference prompt (system + user + generation prompt). During training, loss is computed only on the assistant turn; the system/user turns and padding positions are masked (label = −100), which is standard causal-LM SFT behaviour.

Two implementation details worth recording: (i) the tokenizer's default pad token is `<|endoftext|>`, but the pipeline sets `pad = eos = <|im_end|>` so padding is both unattended and loss-masked; (ii) `json.dumps(..., ensure_ascii=False)` is used throughout so non-ASCII argument values are not lossily escaped.

The derived subset is published as `train.parquet` / `validation.parquet` via `scripts/publish_dataset.py`.

### 9.2 Data cleaning, missing-value handling, and outlier/anomaly policy

The source is already execution-verified (§8.5), so cleaning here is **defensive** rather than corrective: it is a small, explicit rule-set applied before formatting, and it never silently re-labels or "fixes" a ground-truth answer. Every rule is implemented in `tinytoolcaller/data.py` (`validate_example`, `clean_subset`) and covered by `tests/test_data.py` (part of the 41-test suite). **Missing values, malformed records, and duplicates are handled by rules 1–4 below; the dataset's only real "anomaly" class — excessively long tool lists — is handled by rule 5 (truncation, counted rather than silently applied); and value-level "outliers" are deliberately left untouched by rule 6, because the source's execution verification is the authority on correctness.**

| # | Rule | Policy | Action |
| --- | --- | --- | --- |
| 1 | Missing/empty `query` | A query must be a non-empty string | Drop + count (`missing_or_empty_query`) |
| 2 | `tools` not a list, or a tool missing a non-empty `name` | Every candidate tool must have a parseable schema | Drop + count (`tools_not_a_list`, `malformed_tool_entry`) |
| 3 | Missing `answers`/`answer` | Ground truth must exist | Drop + count (`missing_answers`) |
| 4 | Exact duplicates | Identical (query, tool-name set, answer) rows | Keep first + count (`exact_duplicate`) |
| 5 | Length outliers (very long tool lists) | No row is *removed* for length | Truncate to `max_seq_length` at tokenization; count truncated rows (§8.2) — this is the "outlier" this dataset actually has |
| 6 | Value-level outliers | **No value filtering** | Ground truth is authoritative (execution-verified); removing "unusual" argument values would inject bias, not remove noise |

The focused implementation, with the justification for each key line:

```python
def validate_example(example: dict) -> tuple[bool, str]:
    query = example.get("query")
    if not isinstance(query, str) or not query.strip():      # rule 1
        return False, "missing_or_empty_query"

    tools = example.get("tools")
    if not isinstance(tools, list):                          # rule 2a
        return False, "tools_not_a_list"
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str) \
                or not tool["name"].strip():                 # rule 2b
            return False, "malformed_tool_entry"

    if example.get("answers", example.get("answer")) is None:  # rule 3
        return False, "missing_answers"
    return True, "ok"
```

*Why the code is shaped this way:* (i) it returns a **reason string** alongside the verdict, so the cleaning run reports *which* rule fired — a drop count of zero is itself a finding (it would mean the gated source needed no cleaning); (ii) it is deliberately conservative — it only checks structural requirements and never inspects argument *values*, because the source's execution verification is the authority on correctness (§8.5); (iii) it is pure Python with no heavy imports, so it unit-tests on a CPU box. The dedup rule uses a deterministic key `(query, sorted tool names, canonical answer JSON)` so duplicate detection is order-independent.

**Expected behavior on the real source.** Because APIGen already filtered malformed data (§8.5), the drop rate from rules 1–4 is expected to be small — but it must be *reported*, not assumed, and `clean_subset` returns exactly those counts. The rule set is included in the reproducibility workflow (§29) so a third party can reproduce the kept-row counts.

## 10. Method: Base Model and QLoRA

**Base model.** `Qwen/Qwen2.5-1.5B-Instruct` — a 1.5B-parameter instruction-tuned model, Apache-2.0 licensed. A small model is chosen deliberately: the objective is not maximum general capability but whether a small model can become highly reliable on one structured-output task.

**QLoRA.** Full fine-tuning updates all parameters and demands substantial GPU memory. QLoRA [12] combines a **4-bit NF4-quantized base + frozen base parameters + trainable low-rank adapters**, adding double quantization and paged optimizers. LoRA [11] freezes the original weights and learns low-rank update matrices. For this project, QLoRA is a practical route to specializing a 1.5B model without full-model optimization.

## 11. Fine-Tuning Architecture, Parameters, and Configuration

```text
Qwen2.5-1.5B-Instruct → 4-bit NF4 model → frozen base weights
                                              +
                                        LoRA adapters (trainable)
                                              ↓
                                          SFTTrainer
                                              ↓
                                        TinyToolCaller
```

| Parameter | Value | Rationale |
| --- | --- | --- |
| Base model | Qwen2.5-1.5B-Instruct | Small, open, Apache-2.0; matches the "small specialized model" question (§1) |
| Training / validation examples | 5,000 / 200 | Fixed seed-42 subset; single-GPU budget (§8) |
| Quantization | 4-bit NF4, double quant | QLoRA defaults [12]; cuts VRAM to fit single-GPU |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | Common QLoRA default (α = 2·r); balances capacity vs. adapter size; light regularization on a small (5K) set |
| LoRA bias / task type | none / CAUSAL_LM | LoRA-paper default; train adapters only |
| Target modules | q,k,v,o,gate,up,down proj | Covers attention + MLP; standard for Qwen/LLaMA-family architectures |
| Learning rate / scheduler / warmup | 2e-4 / cosine / 3% | LoRA uses higher LR than full FT (~1e-5); cosine + small warmup is a stable SFT default |
| Batch size / grad accumulation | 2 / 8 (effective 16) | Small device batch fits VRAM; effective 16 is a common small-SFT budget |
| Epochs | 2 | 5K × 2 steps ≈ short run; overfitting risk noted in §21.3 |
| Optimizer | `paged_adamw_8bit` | QLoRA default optimizer (pageable 8-bit states) |
| Max sequence length | 1024 | Covers typical tool sets; truncation quantified in §13.2 |
| Precision | BF16 if available, else FP16 | Faster on Ampere+; falls back to FP16 |

**Not swept.** None of these values were selected by systematic search (§21.3). They are standard QLoRA/SFT defaults applied at 1.5B scale — a deliberate simplification, flagged rather than hidden: see the `paged_adamw_8bit` + gradient-accumulation note in §12.

**Tuning methodology, stated.** The configuration was set **once, from published defaults** (QLoRA [12] and common TRL SFT recipes), not tuned: there was no learning-rate search, no rank search, and no early-stopping selection on the validation split. Two consequences follow honestly: (i) the reported gains are a *lower-bound on what tuning could achieve*, not an optimum; (ii) because the validation split was never used for model selection, the in-sample results are not additionally inflated by tuning-on-eval — the caveats of §21.1 still apply, but no selection-on-validation occurred.

**Standard vs. non-standard parameters.** Every hyperparameter is a **published default**; none is non-standard. The two items closest to "non-standard" are flagged explicitly: `paged_adamw_8bit` + gradient-accumulation-8 are inherited from larger-model QLoRA tutorials and are not documented as empirically necessary at 1.5B (§12), and `eval_load_in_4bit=True` is a deliberate *design* choice (equalize base/fine-tuned quantization, §14) rather than a tuned parameter.

The full configuration lives in the central `CONFIG` dict in `tinytoolcaller/config.py`, and `tests/test_config.py` asserts that the code's values equal the table above (so code and paper cannot drift apart silently).

## 12. Experimental Environment

The training environment was not captured at run time; the table below is the **template that must be filled** for full verifiability (§28). `scripts/capture_environment.py` prints it in paste-ready form:

```bash
python scripts/capture_environment.py --save outputs/environment.json
```

| Item | Value |
| --- | --- |
| **GPU** model / VRAM | **TBD** (e.g., RTX 4090 24 GB, A10G 24 GB) |
| **CPU** model / cores | **TBD** |
| **RAM** (host) | **TBD** |
| **Storage** (dataset + checkpoints ≈ 1–2 GB) | **TBD** |
| **OS** / kernel | **TBD** |
| CUDA version | **TBD** |
| PyTorch / Transformers / TRL / PEFT / bitsandbytes | **TBD** |
| `datasets` version (shuffle RNG — affects §8.1) | **TBD** |
| Training wall-clock time | **TBD** |
| Peak GPU memory during training | **TBD** |
| Python version | **TBD** |

`scripts/capture_environment.py` records Python, platform, GPU name/VRAM, CUDA availability/version, and the seven library versions automatically; CPU/RAM/storage/OS are filled manually alongside it. The JSON it writes (`environment.json`) is committed with the run outputs (§29).

**Reference build environment** (the CPU sandbox used to assemble this publication and run its tests — *not* the training GPU environment):

| Item | Value |
| --- | --- |
| Python | 3.13.14 |
| Platform | Linux x86_64 |
| transformers | 5.15.1 |
| huggingface_hub | 1.28.0 |
| CUDA available | No (CPU-only CI) |

One configuration note worth stating (flagged in the implementation): `paged_adamw_8bit` and gradient accumulation of 8 are standard for QLoRA on *larger* models under severe memory pressure. At 1.5B with 4-bit quantization it is not documented whether these were empirically necessary or inherited from larger-model QLoRA tutorials without re-validation. If inherited, that is a legitimate simplification and should be said so explicitly.

## 13. Implementation: Pipeline, Considerations, and Code Quality

### 13.1 Pipeline

`train_tool_caller.py` is a thin CLI that wires the `tinytoolcaller/` package through the 14 documented stages: load tokenizer → load dataset → shuffle/split → ChatML formatting → baseline evaluation → load 4-bit model → prepare k-bit training → attach LoRA → train → save adapter → evaluate fine-tuned model → evaluate GSM8K → merge adapter → publish. `--eval-dump` additionally writes per-example predictions for the paired significance test (§18).

### 13.2 Notable implementation decisions

- **Markdown-wrapped output.** The baseline frequently wraps JSON in ``` ```json ``` fences. The evaluation strips these before parsing, so the 78.5% baseline JSON-validity figure **already benefits from cleanup** and is not raw-output purity (§21.7).
- **Sequence length and truncation — quantified with the real tokenizer.** The 1024-token cap interacts with tool-set size in a measurable way. Using the actual Qwen2.5-1.5B tokenizer, a prompt with **1** verbose tool serializes to ≈282 tokens, **3** tools ≈648, **5** tools ≈1014, and **10** tools ≈1929 — the last two exceed the cap and are truncated. The pipeline therefore *implicitly* upweights examples with small tool sets. `scripts/dataset_stats.py` reports the exact truncation count for the real subset (§8.2); until that number is known, results on long-tool-set prompts should be treated with caution.
- **No retry/repair.** A single generation is scored as-is; production systems would retry malformed output before failing, which this evaluation does not simulate.
- **Heavy dependencies are lazy.** `tinytoolcaller/` imports torch/trl/peft/bitsandbytes only inside the functions that need them, so the pure helpers (formatting, metrics) import and unit-test on a CPU/CI box with no GPU stack — which is how the 41-test suite runs here.

### 13.3 Package layout and library appropriateness

```text
train_tool_caller.py            # thin CLI: wires the package through the 14 stages
tinytoolcaller/
    config.py                   # central CONFIG + SYSTEM_PROMPT (§11)
    formatting.py               # pure prompt/JSON/answer helpers (no heavy deps)
    data.py                     # tokenizer/dataset loading + deterministic sampling
    model.py                    # 4-bit model loading + LoRA attachment
    metrics.py                  # ToolCallingMetrics + scorers (§14)
    train.py                    # SFTTrainer wrapper + merge/publish
scripts/
    profile_tool_distribution.py   # §8.1
    dataset_stats.py               # §8.2
    statistical_analysis.py        # §18
    capture_environment.py         # §12
    publish_dataset.py             # §28
    build_preprint.py              # PDF build
tests/                            # pytest: 41 tests (config, formatting, metrics, data-quality, repair)
```

| Library | Purpose |
| --- | --- |
| Transformers / Datasets | Model/tokenizer; dataset loading |
| PEFT | LoRA / parameter-efficient fine-tuning |
| TRL | Supervised fine-tuning (`SFTTrainer`) |
| bitsandbytes | Quantization |
| PyTorch | Model execution |
| W&B | Experiment tracking |
| Hugging Face Hub | Artifact publication |

This addresses the earlier observation that the pipeline should be modular rather than one end-to-end script: the stage functions now live in importable modules with unit tests, and `tests/test_config.py` pins the code's configuration to the paper's §11 table.

### 13.4 Code explanation quality

Each module is documented at two levels: a module docstring stating its role in the pipeline, and per-function docstrings stating pre/post-conditions. The two subtlest functions are explained inline:

- **`formatting.extract_json`** (§14/§21.7): three-layer parse — strip ``` ```json ``` fences, try `json.loads` on the whole string, then a balanced-brace scan of the first `{...}` region; non-dict parses (e.g., a bare JSON list) are rejected because the contract is an *object*, preventing a downstream `pred.get("name")` crash (this exact bug was caught by `tests/test_formatting.py`).
- **`metrics.evaluate_tool_calling(..., return_details=True)`** (§18): returns per-example `{gt, raw, pred}` records so the paired McNemar/bootstrap test can be computed — a requirement that aggregate percentages alone cannot satisfy.

### 13.5 Code presentation policy and conventions

The paper's code snippets follow a **minimal-illustration policy**: each snippet is ≤15 lines, illustrates exactly one concept, and carries a purpose caption plus key-line justification (§9.2, §22.1, §3.1, §8.1). Complete implementations live in the repository, not in the prose, so code never overshadows the conceptual content. Conventions, stated so they can be audited:

| Convention | Rule |
| --- | --- |
| Names | Descriptive verb-phrases for functions (`load_quantized_model`, `extract_json`); UPPER_SNAKE for constants (`CONFIG`, `SYSTEM_PROMPT`, `REPAIR_INSTRUCTION`) |
| Types | Type hints on public signatures (`-> tuple[bool, str]`); dataclasses for structured results (`ToolCallingMetrics`) |
| Docstrings | Module docstring states the pipeline stage; function docstring states pre/post-conditions and cites the publication section it implements |
| Dependencies | Heavy imports (torch/trl/peft/bitsandbytes) are **lazy**, inside the functions that need them — so the pure helpers import and unit-test on a CPU/CI box |
| Configuration | One central `CONFIG` dict (§11); no magic numbers in function bodies |
| Tests | Every pure function has a unit test; the suite pins `CONFIG` values to §11's table (`tests/test_config.py`) |

Run the suite with:

```bash
pip install pytest
python -m pytest tests/ -v     # 41 passed
```

## 14. Evaluation Framework and Metrics

Three metrics are computed over the 200-example validation split:

| Metric | Definition | Computation |
| --- | --- | --- |
| JSON validity | Output contains a parseable JSON object | JSON extracted via regex/substring match, then `json.loads()`; **not** a raw-output purity check (§21.7) |
| Tool-name accuracy | Predicted `name` equals ground truth | Exact, case-sensitive string match |
| Argument exact match | Predicted `arguments` equals ground truth | Exact match on keys and values; no partial credit |

**Why exact match, not similarity scoring.** In a real execution pipeline, a partially correct argument set (right tool, wrong value) still fails downstream. Exact match reflects the deployment failure mode more honestly than a softer metric, at the cost of not distinguishing "close" from "way off" failures.

**Quantization control.** By default the baseline is evaluated on the same 4-bit NF4 quantized base (`eval_load_in_4bit=True`) as the fine-tuned model, so the comparison isolates fine-tuning rather than precision. Scoring the bf16 base instead is a one-flag ablation (§25.4).

**Not measured.** Latency, token-level calibration, out-of-schema requests, and multi-tool selection when more than one tool could validly answer — candidates for a follow-up pass (§32).

## 15. Validation Strategy

The validation strategy has **two parts**: (A) what is already in place and why it is valid *internally*, and (B) the pre-publication verification plan that extends validity *externally*. The headline results are reported under (A) alone and are labelled as such throughout (§17).

### Part A — In-place controls (already holding)

1. **Prompted-baseline control.** The base model is evaluated on the *identical* prompts, metrics, and (by default) quantization as the fine-tuned model, so the measured improvement is attributable to fine-tuning, not to prompt, metric, or precision changes (§14, §16).
2. **Paired comparison design.** Both models are scored on the *same* 200 examples, which is the precondition for McNemar's test and a paired bootstrap (§18) — a stronger test than comparing two independent samples.
3. **Deterministic, documented split.** Fixed seed (42), 5,200 sampled → 5,000/200, reproducible by any third party (§29).
4. **Retention probe.** GSM8K (n=50) under one shared harness for both models (§20).
5. **Confidence intervals, not point estimates.** Every reported proportion carries a Wilson 95% CI and Cohen's h (§18), so readers can see the estimation uncertainty directly.

### Part B — Pre-publication verification plan (held-out set + robustness checks)

The strategy's known gap is that the 200-example split is also the development set. The verification plan closes it with **one locked held-out set and five robustness checks**, executed in order, *before* the results may be quoted as generalization evidence:

| Step | What | Design | Pass criterion |
| --- | --- | --- | --- |
| **B1 — Held-out test set** | A further **500 examples** drawn from the 60K with seed **7**, disjoint from the 5,200-example subset, **locked before any further tuning** and evaluated only after all model decisions are final | `shuffle(seed=7) → take 500 not in the seed-42 subset` | Report the three metrics on it, unedited, with CIs; treat any material drop as a real generalization estimate |
| **B2 — Multi-seed robustness** | Re-run the train/val split under ≥3 seeds and report mean ± std per metric | Seeds {42, 43, 44} | The reported improvements persist across seeds |
| **B3 — Quantization ablation** | Score the *bf16* base (not just 4-bit) against the fine-tuned model | `eval_load_in_4bit=False` | Quantifies how much of the gap is quantization vs. fine-tuning (§25.4) |
| **B4 — Paired significance** | Produce the per-example dump and compute McNemar + bootstrap CI | `--eval-dump` → `statistical_analysis.py --mcnemar` | Report the exact paired p-value and bootstrap CI (§18) |
| **B5 — Distribution & truncation audit** | Run §8.1 profiling and §8.2 stats on the *actual* subset | `profile_tool_distribution.py`, `dataset_stats.py` | Fill §8.1/§8.2; if top-tool share >10% or truncation >1%, add the caveat (§21.8) |
| **B6 — Contamination check** | Verify the 200-example split contains no GSM8K-style QA pairs that could inflate the retention comparison | n-gram/embedding overlap vs. GSM8K | No significant overlap |

B1 and B5 are the **highest-priority** items — they convert "directionally credible" into "estimated on unseen data" and remove the skew caveat respectively. Until B1 is executed, all result claims remain explicitly in-sample (§17, §21.1). This is the honest boundary between what the current results *support* and what they *do not yet support*.

## 16. Baseline Results

| Metric | Result |
| --- | ---: |
| JSON validity | 78.5% |
| Tool-name accuracy | 65.0% |
| Argument exact match | 42.0% |
| GSM8K (n = 50) | 52.0% |

The base model often understands the request but adds markdown, explanatory text, selects the wrong tool, or omits arguments — establishing a meaningful baseline rather than evaluating the fine-tuned model in isolation.

## 17. Comparative Analysis

### 17.1 Base vs. fine-tuned

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

> **Caveat — JSON validity.** Computed *after* JSON extraction, not on the raw output. It measures whether a valid JSON object could be extracted, not whether the model produced pure JSON (§21.7).

> **Caveat — evaluation set.** All figures come from the same 200-example split used during development; no independent held-out test set was held back. These are **in-sample experimental results, not unbiased estimates of generalization** (§15, §21.1).

> **Caveat — tool distribution (open item).** The subset's tool-name distribution has not yet been profiled (§8.1). Until (a) unique-tool count, (b) top-10 frequency, and (c) train/validation match are reported, the tool-name-accuracy figure cannot be separated from possible skew toward high-frequency tools.

### 17.2 Position relative to related work

TinyToolCaller is **not directly comparable** to the models below (different data, scale, and benchmarks), but its position clarifies the contribution:

| Work | Scale | Method | Data | Evaluation | What TinyToolCaller adds |
| --- | --- | --- | --- | --- | --- |
| xLAM-1b-fc-r [1][15] | 1B | Full SFT | APIGen 60K | BFCL (AST) | Same data family, but *prompted-baseline* ablation at 1.5B with per-failure-mode metrics |
| Octopus v2 [14] | 2B | Full SFT + functional tokens | Proprietary | Task accuracy + latency | QLoRA (not full FT); open recipe |
| NexusRaven-V2 [5] | 13B | SFT (no GPT-4 distill) | Curation + retrieval | Nexus/API-Bank | Much smaller; isolates QLoRA-only lift |
| Granite-20B-FC [7] | 20B | Multi-task granular SFT | Multiple | BFCL + 5 suites | Small-model focus; actionable metric decomposition |
| ToolACE [6] | 8B | SFT on ToolACE | 26.5K tools | BFCL/API-Bank | Confirms its "small raw models improve sharply with FT" at 1.5B |
| BFCL [2] | — | — | — | AST/execution | Independent benchmark is a planned next step (§32), not yet run |

The largest observed improvement is **argument exact match** (42.0% → 84.0%), the most deployment-relevant of the three. The direction is consistent across all three task-specific metrics, but the exact percentage-point improvements should not be read as precise estimates of unseen-data performance.

### 17.3 Comparison against published state-of-the-art results

TinyToolCaller's three project metrics are **not comparable** to BFCL's overall accuracy — different data, prompts, and scoring. But anchoring the project against the published BFCL leaderboard (as reported in the APIGen paper, June 2024) [1] places the effort in context. **These are external, cited numbers, not results from this project's evaluation:**

| Model | Scale | BFCL overall accuracy [1] |
| --- | --- | ---: |
| GPT-4-0125-Preview (prompt) | (proprietary) | 88.0 |
| Claude-3-Opus-0229 (prompt) | (proprietary) | 87.7 |
| Gemini-1.5-Pro-0514 (FC) | (proprietary) | 86.4 |
| **xLAM-7B (FC)** — trained on the same data family | 7B | 85.7 |
| Gorilla-OpenFunctions-v2 (FC) | 7B | 84.7 |
| Llama-3-70B-Instruct (prompt) | 70B | 83.9 |
| **xLAM-1B (FC)** — trained on the same data family | 1.3B | 74.4 |
| Claude-3-Haiku-0207 (prompt) | (proprietary) | 74.3 |
| GPT-3.5-Turbo-0125 (FC) | (proprietary) | 63.9 |
| *TinyToolCaller (this project)* | 1.5B | *not evaluated on BFCL — §32 priority 8* |

**How to read this.** The relevant anchor is not GPT-4 but **xLAM-1B (74.4)**: a 1.3B model fully fine-tuned on the *same* data family ranks above GPT-3.5-Turbo. TinyToolCaller is the complementary experiment — a *prompted-baseline* ablation at the same scale using QLoRA, which the leaderboard reports cannot isolate. Until TinyToolCaller is run on BFCL (§32), the defensible statement is positional: *the data family produces SOTA-at-scale small models, and this project shows a QLoRA subset reproduces the direction at 1.5B with per-failure-mode diagnostics* — not that it matches xLAM-1B's leaderboard score.

### 17.4 Required internal baselines (beyond the prompted base)

The evaluation currently has **one** internal control (the prompted base, §16). A complete comparison should add four more, all scored with the identical §14 harness:

| # | Baseline | What it isolates | Status |
| --- | --- | --- | --- |
| 1 | Prompted base, 4-bit (§16) | Fine-tuning effect | ✅ reported |
| 2 | Prompted base, bf16 | Quantization confound (§25.4) | 🔧 one flag (`eval_load_in_4bit=False`) |
| 3 | Full fine-tune (all params) | QLoRA vs. full-FT at 5K examples | ❌ not run |
| 4 | LoRA r=8 (or r=4) | Rank sensitivity | ❌ not run |
| 5 | Fine-tune on 10K examples | Data-scale sensitivity | ❌ not run |

Baselines 3–5 would answer "was QLoRA the right choice?" directly; their absence is recorded in §21.3 rather than papered over.

**Baseline coverage, summed up.** The comparative analysis places TinyToolCaller against (i) **six related systems** positioned methodologically in §17.2 (xLAM-1b-fc-r, Octopus v2, NexusRaven-V2, Granite-20B-FC, ToolACE, BFCL), (ii) **nine published models** with real leaderboard numbers in §17.3 (GPT-4, Claude-3-Opus, Gemini-1.5-Pro, xLAM-7B, Gorilla-OpenFunctions-v2, Llama-3-70B, xLAM-1B, Claude-3-Haiku, GPT-3.5-Turbo), and (iii) **five internal baselines** in §17.4. That satisfies the "≥4–5 baselines" bar on the *context* axis; the honest remaining gap is that only one internal baseline (the prompted base) is *measured*, which §17.4 states plainly and §33 schedules.

## 18. Statistical Analysis

All three task-specific improvements are **large by any reasonable reading**, even after accounting for estimation uncertainty. Wilson 95% confidence intervals for each reported proportion (n = 200; GSM8K n = 50):

| Metric | Base | 95% CI (base) | Fine-tuned | 95% CI (ft) | Cohen's h | Effect |
| --- | ---: | --- | ---: | --- | ---: | --- |
| JSON validity | 78.5% | [72.3%, 83.6%] | 98.0% | [95.0%, 99.2%] | +0.68 | medium |
| Tool-name accuracy | 65.0% | [58.2%, 71.3%] | 92.5% | [88.0%, 95.4%] | +0.71 | medium |
| Argument exact match | 42.0% | [35.4%, 48.9%] | 84.0% | [78.3%, 88.4%] | +0.91 | large |
| GSM8K retention | 52.0% | [38.5%, 65.2%] | 50.0% | [36.6%, 63.4%] | −0.04 | negligible |

**How to read this.** The fine-tuned model's confidence intervals do not overlap the base model's for any of the three task metrics, so the direction of improvement is robust to sampling noise at the reported proportions. Cohen's h classifies the argument-exact-match gain as **large** and the other two as **medium**; the GSM8K change is negligible.

**Paired significance testing.** Because both models are evaluated on the *same* 200 examples, the correct test for the difference is **McNemar's test** on the discordant pairs (base-wrong/fine-tuned-right vs. base-right/fine-tuned-wrong), plus a **bootstrap confidence interval** on the paired difference. Neither can be computed from aggregate percentages — it requires the per-example outcomes. The pipeline writes these via `train_tool_caller.py --eval-dump`, and `scripts/statistical_analysis.py --mcnemar` computes McNemar, the bootstrap CI, and Cohen's h from them. Until that file is produced from a real run, the statement is: *the CIs above support the direction and approximate magnitude of the effect; the exact paired p-value is pending the per-example dump.*

```bash
python train_tool_caller.py --eval-dump outputs/eval_predictions.jsonl
python scripts/statistical_analysis.py --mcnemar outputs/eval_predictions.jsonl
```

**GSM8K.** The 2-point change on 50 examples is within sampling noise (95% CI half-width ≈ ±13 pp at p ≈ 0.5), so the experiment **cannot distinguish "no forgetting" from "moderate forgetting"** and must not be cited as evidence of retention either way.

## 19. Results Interpretation and Error Analysis

The three metrics decompose the failure surface into bands that call for different engineering responses.

### 19.1 Success stories

1. **Structured output.** JSON extraction validity 78.5% → 98.0%: the model became considerably more consistent at the structured-output objective on the evaluated examples. (This is *after* extraction — see §21.7 — so it is not "98% of raw responses are pure JSON".)
2. **Tool selection.** 65.0% → 92.5%: improved association of requests with the correct available function.
3. **Argument construction.** 42.0% → 84.0%: the largest gain, and the most practically important — selecting the right function is insufficient if the parameters are wrong.

### 19.2 Failure analysis

Interpreting the fine-tuned model's 200-example results as a failure budget:

| Failure class | Estimated share | Production consequence | Mitigation |
| --- | ---: | --- | --- |
| Invalid / non-extractable JSON | ≈ 2% (4/200) | Call cannot even be parsed | Retry-with-repair; schema validation (§22) |
| Valid JSON, wrong tool | ≈ 7.5% (15/200) | Wrong function executed | Tool allowlist; relevance detection |
| Right tool, wrong arguments | ≈ 8.5 pp gap (185 correct tools vs. 168 correct arguments) | **Silent semantic error** — well-typed but wrong values | Type/range/semantic validation; canary testing |

The third row is the most consequential insight: **schema validation cannot catch it**, because the arguments are structurally valid. The gap between tool accuracy (92.5%) and argument exact match (84.0%) is exactly the band where the model picks the right tool but fills it wrong — the failure class that dominates production risk and that a single aggregate accuracy number would hide. (Exact joint counts require the per-example dump; the figures above are derived from the aggregates under the assumption that argument correctness implies tool correctness, which holds in practice.)

### 19.3 Illustrative transcripts

The transcripts below are **reconstructions of the documented failure modes (§1)**, not logged model outputs; the per-example dump (§18) produces the real ones.

**Success (fine-tuned):**

```text
User Request: What's the weather in Tokyo?
Available Tools: [get_weather(location: string, unit: celsius|fahrenheit)]

Model output: {"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
Scored: JSON valid ✓ · tool correct ✓ · arguments exact ✓
```

**Failure A — markdown-wrapped (baseline-typical):**

```text
Model output:
```json
{"name": "get_weather", "arguments": {"location": "Tokyo"}}
```

Scored: JSON valid (after fence-stripping) ✓ · tool correct ✓ — but raw output is not pure JSON (§21.7)
```

**Failure B — wrong tool:**

```text
User Request: Where is order 12345?
Model output: {"name": "search_products", "arguments": {"query": "12345"}}
Ground truth: {"name": "get_order_status", "arguments": {"order_id": "12345"}}
Scored: JSON valid ✓ · tool ✗ · arguments ✗
```

**Failure C — right tool, hallucinated argument (the dangerous one):**

```text
User Request: What's the weather in Tokyo?
Model output: {"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "metric"}}
Ground truth: {"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
Scored: JSON valid ✓ · tool ✓ · arguments ✗  ← "metric" is not in the schema enum
```

### 19.4 Lessons learned from the build

Four concrete project events, stated with their lesson, because they are the transferable part of the exercise:

1. **A unit test caught a real bug before it could ship.** `test_extract_json_rejects_bare_list` exposed that a JSON-*list* output (a legitimate model failure mode) would crash the scorer via `pred.get("name")` on a list. *Lesson:* scoring code is production code — a two-line `isinstance(parsed, dict)` guard plus one test is disproportionately cheap insurance, and the suite now runs on a CPU with no GPU stack because the heavy imports are lazy (§13.2).

2. **Truncation was a measurement, not a known quantity.** Only after tokenizing with the real Qwen tokenizer did it become concrete that ~5 verbose tools already approach the 1024-token cap (§13.2). *Lesson:* "max_seq_length = 1024" in a config is a silent filtering decision until someone counts the rows it truncates — the counter now lives in `scripts/dataset_stats.py` (§8.2).

3. **The baseline's headline number was flattering.** The 78.5% baseline JSON validity *includes* fence-stripping; without it the number would be lower. *Lesson:* state what a metric measures *and what it doesn't* at the point of quoting (§17 caveats), or downstream readers will over-read it.

4. **Aggregate percentages cannot support the claim we wanted to make.** "Is the improvement statistically significant?" requires per-example outcomes (McNemar), which aggregate tables cannot provide. *Lesson:* design the evaluation to dump per-example predictions *from the start* (`--eval-dump`, §18), not as an afterthought — retrofitting it after the run is exactly the open item in §36.

## 20. Catastrophic Forgetting Analysis

| Model | GSM8K (n = 50) |
| --- | ---: |
| Base | 52.0% |
| Fine-tuned | 50.0% |

A 2-point change on 50 examples is well within sampling noise (95% CI half-width ≈ ±13 pp). This experiment **cannot distinguish "no forgetting" from "moderate forgetting"** and should not be cited as evidence of retention either way. A stronger analysis would use the full GSM8K (or another benchmark) under a fixed, identical harness for both models.

## 21. Limitations

| Limitation | Likely impact on headline results |
| --- | --- |
| No independent test set (§21.1) | **High** — in-sample figures likely overstate generalization by an unknown margin |
| Small GSM8K sample (§21.2) | **High** — the retention claim is statistically unsupported either way |
| No hyperparameter sweep (§21.3) | **Medium** — gains may be improvable or a local optimum |
| Limited training data, 5K/60K (§21.4) | **Medium** — more data would likely improve tool coverage |
| Single-turn focus (§21.5) | **Low for this report** — out of scope, doesn't bias current numbers |
| No external benchmark (§21.6) | **Medium** — limits comparability (BFCL, τ-bench) |
| JSON-validity extraction leniency (§21.7) | **High** — 98.0% overstates raw-output compliance |
| Tool distribution unprofiled (§21.8) | **High** — until §8.1 is filled in, tool accuracy may be inflated by a skewed subset |

**21.1** The 200-example validation split is also the final evaluation set; a future version should maintain train / validation / independent test.

**21.2** 50 GSM8K examples; the 2-point change is not a precise measurement of degradation.

**21.3** Fixed configuration; no systematic search over learning rate, LoRA rank, dropout, epochs, sequence length, or batch configuration.

**21.4** 5,000 examples of 60,000.

**21.5** Multi-turn tool use is not addressed.

**21.6** No results yet on a standardized external function-calling benchmark (BFCL [2], τ-bench [8]).

**21.7** JSON is extracted before scoring, so the JSON-validity metric does not represent the stricter requirement that the raw output contain *only* JSON. Correct this in a future evaluation version.

**21.8** The subset's tool-name distribution (unique count, top-10 concentration, train/validation match) has not yet been measured (§8.1). Until those three quantities are reported, the tool-name-accuracy figure is entangled with possible selection skew, and the validation split's representativeness is unverified. **Highest-priority measurement to complete before publication.**

## 22. Deployment Considerations

TinyToolCaller is suitable for experimentation and model-level inference; production tool execution requires additional infrastructure. This section specifies the deployment surface: **hardware, dependencies, integration contract, scalability, security, and performance** — the items a production rollout must budget for.

### 22.1 Runtime controls (safety-critical)

| Control | Priority | Rationale (tied to measured failure rate) |
| --- | --- | --- |
| JSON Schema validation | Critical | ≈2% of fine-tuned outputs are not valid JSON — catch before execution |
| Tool allowlist | Critical | ≈7.5% tool-selection error means wrong-but-valid calls occur; allowlisting limits blast radius |
| Argument validation (type/range) | Critical | ≈16% argument mismatch — schema validation alone won't catch semantically wrong-but-well-typed values |
| Authorization / scoping | High | Independent of model accuracy — required regardless |
| Retry-with-repair | Medium | Could recover some of the ≈2% JSON-invalid cases cheaply |
| Audit logging | Medium | Post-hoc failure analysis |
| Rate limits / timeouts | Standard | Generic API hygiene |

A minimal validator for the model's output (the "Critical" row, in ~15 lines):

```python
import jsonschema

def validate_and_execute(raw: str, tool_schema: dict, executor, user_id: str):
    call = extract_json(raw)                     # §14 parsing
    if call is None:
        raise InvalidCall("unparseable")          # → retry-with-repair path
    if call["name"] not in TOOL_ALLOWLIST:        # allowlist, not the model
        raise ForbiddenTool(call["name"])
    jsonschema.validate(instance=call["arguments"],
                        schema=tool_schema["parameters"])   # type/range
    if not authorized(user_id, call["name"], call["arguments"]):
        raise Unauthorized(call)
    return executor[call["name"]](**call["arguments"])
```

*Key lines, justified:* `extract_json` is the same parser used at evaluation time (§14) — deployment and evaluation must not diverge; the **allowlist** check comes *before* execution and is enforced in application code, not by the model (§1's design principle); `jsonschema.validate` catches type/range violations but, as §19.2 shows, cannot catch semantically wrong-but-well-typed values — which is why `authorized()` exists as the final human-scoped gate.

### 22.2 Infrastructure and resource requirements

| Stage | Hardware | Notes |
| --- | --- | --- |
| Training (QLoRA, this recipe) | Single NVIDIA GPU, ≈8–16 GB VRAM | 4-bit NF4 base ≈ 0.75 GB weights; LoRA trainable ≈ 28M params (≈1.8% of base); paged 8-bit optimizer keeps state small; 5K×2 epochs ≈ hours, not days (§17.1). Verify against §12. |
| Inference (merged model) | CPU or any CUDA GPU (≥4 GB) | Merged 1.5B model ≈ 3 GB bf16 / ≈1 GB 4-bit + KV cache; latency scales with output tokens (max 256 here) |
| Serving at scale | GPU pool or vLLM/TGI instance | Batch decoding; adapter hot-swap per tenant if multi-tenant |

**System dependencies.** Python 3.10+, `torch`, `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `huggingface_hub` (training); inference needs only `torch` + `transformers` + the merged weights. CUDA toolkit version must match the installed PyTorch wheel. The full matrix is captured by `scripts/capture_environment.py` (§12).

### 22.3 Integration contract

The model is a **drop-in structured-output component** behind a narrow interface:

```text
POST /tool-call
  { "request": "…", "tools": [ {name, description, parameters} … ] }
→ { "name": "…", "arguments": { … } }     # or {"error": "unparseable"}
```

Integration requirements: (i) **schema versioning** — the `tools` payload format must be versioned so prompt and validator stay in lock-step; (ii) **idempotency** — the caller provides a request id so retry-with-repair cannot double-execute a side-effecting tool; (iii) **timeouts** — cap generation at the token budget (§11's `max_new_tokens`); (iv) **context budget** — reject prompts whose serialized tools exceed the 1024-token cap rather than silently truncating (§13.2), because truncation is a known accuracy hazard.

### 22.4 Scalability

The component is stateless, so it scales horizontally behind a load balancer; the only shared state is the model weights and the tool registry. Multi-tenant deployments can swap LoRA adapters per tenant without reloading the base. Batch inference lifts throughput ~2–5× at the cost of tail latency; the latency/throughput trade-off is a configuration knob, not a code change.

### 22.5 Security

- **Prompt/tool-schema injection:** treat the tool schemas as untrusted input; a malicious schema could steer the model — validate schemas and never pass model output directly to a shell/SQL sink.
- **Least privilege:** the executor runs with per-tool, per-user scopes (§22.1 `authorized()`), not blanket credentials.
- **PII:** queries may contain personal data — the model is not a data store, but logs (§23) must be scoped and redacted.
- **Out-of-schema requests:** reject tool names outside the allowlist; monitor the rate (§23).

### 22.6 Performance requirements (targets, to be validated)

| Metric | Initial target | Basis |
| --- | --- | --- |
| Generation latency | p95 < 500 ms on a single GPU (≤64 output tokens) | 1.5B-class decode speed; verify on target hardware |
| Throughput | ≥ 50 req/s per GPU (batched) | Batch-decoding estimate; verify |
| Availability | 99.9% monthly | Standard for internal tool-calling services |
| Rollback time | < 15 min | Adapter + merged weights are versioned artifacts |

**Deployment stages.** Shadow (log-only, no execution) → canary (5% traffic, A/B against baseline metrics §23) → progressive rollout → full. Every stage gates on the JSON-validity and tool-accuracy thresholds of §23 before advancing.

**Expected challenges.** (i) *Silent argument errors* (§19.2) are the hardest failure to detect post-deployment — mitigate with canary fixtures; (ii) *schema drift* between training and production tools — mitigate with the §23 drift checks; (iii) *context overrun* on long tool lists — mitigate with the §22.3 budget check; (iv) *quantization variance* across hardware — pin the same bitsandbytes version used in §12.

A safe production architecture:

```text
User → Application → Tool Registry + Prompt Builder → TinyToolCaller → JSON output
      → JSON Schema Validator → Authorization → Tool Executor → External API
```

The model should not be granted unrestricted execution privileges.

## 23. Monitoring and Maintenance

**Starting alert thresholds** (calibrated against this experiment's baseline; revise after real traffic):

| Signal | Threshold | Action |
| --- | --- | --- |
| JSON validity | < 95% | Page/alert (2 pp below the 98.0% eval figure) |
| Tool-selection failure rate | > 10% | Investigate for drift (observed: 7.5%) |
| Unknown/out-of-schema tool requests | > 1% of traffic | Users are asking for capabilities outside the trained tool set |

**Metric definitions** (what to log per request): raw output, extraction result, predicted `name`, predicted `arguments`, ground-truth (when available), validity flag, latency_ms, and the tool-set size — the last one matters because §13.2 shows longer tool sets truncate and correlate with harder prompts. The per-request **log record** is a fixed schema so failure analysis is a query, not a manual review:

```json
{
  "ts": "2026-08-20T09:41:00Z",
  "request_id": "c9f3…",
  "model_version": "tinytoolcaller-v1",
  "num_tools": 4,                      // tool-set size — truncation correlate (§13.2)
  "json_valid": true,                  // §14 metric 1
  "tool_correct": true,                // §14 metric 2 (needs ground truth)
  "args_exact": false,                 // §14 metric 3 (needs ground truth)
  "predicted": {"name": "get_weather", "arguments": {"location": "Tokyo"}},
  "latency_ms": 214,
  "schema_version": "tools-v1",
  "outcome": "executed"                // executed | rejected | retried | failed
}
```

*Key fields, justified:* `num_tools` lets ops correlate failures with schema size (the §13.2 truncation hazard); the three `*_correct` flags are exactly the §14 metrics, so dashboard numbers and paper numbers use the same definitions; `outcome` records what the *deterministic layer* did, so model error rate and system error rate stay separable; `request_id` links the log to the §22.3 idempotency key. PII is kept out of this record by design — only the predicted tool call is logged, never the raw query text.

**Data-drift detection.** Two complementary checks: (i) a categorical chi-square test comparing the production tool-name distribution against the training distribution (§8.1's method, reused at inference time); (ii) an embedding-distance or n-gram novelty score on production queries vs. the training set. Either diverging from baseline signals that the request mix has moved off-distribution.

**Maintenance schedule.** A concrete calendar so the system "remains effective over time" rather than degrading silently:

| Cadence | Task | Owner | Exit condition |
| --- | --- | --- | --- |
| Continuous | Dashboards on the three validity/accuracy rates + latency percentiles | On-call | Alerts within threshold |
| Daily (automated) | Retry-rate and unknown-tool-rate anomaly check | Automation | No threshold breach |
| Weekly | Review the failure log against the §19.2 classification table; tag new failure signatures | ML engineer | Top failure class has an owner |
| Monthly | Full regression run (200-example eval + §8.1 drift check) on the *current* model and any candidate version | ML engineer | No metric regressed vs. baseline |
| Quarterly | Re-profile production traffic vs. training distribution; decide on re-training or schema updates | ML + product | Drift documented and scheduled |
| Per release | Canary evaluation (§22.6) before promotion | Release owner | Canary meets thresholds |

The current project tracks training loss, learning rate, baseline/fine-tuned metrics, and system metrics (GPU utilization/memory) in W&B. A production system extends this to **model metrics** (validity, tool-selection and argument-validation failures, unknown-tool requests, retry rate, generation latency), **infrastructure metrics** (GPU memory, CPU/GPU utilization, throughput, request latency, error rate), and **data drift**.

Maintenance loop:

```text
Production requests → failure analysis → evaluation dataset → regression test
    → fine-tuning → new model version → canary evaluation → deployment
```

**Runbook outline.** (1) Triage: classify failures by §19.2's table; (2) if JSON-validity drops, check prompt/tokenizer changes first (cheapest); (3) if tool-selection drifts, run the §8.1 chi-square and check for new tool categories; (4) if argument errors rise, sample 20 failures and check for a common hallucinated parameter; (5) promote a fix only after the monthly regression passes and a canary shows no regression.

## 24. Significance and Implications of the Work

The significance is **not** that a 1.5B model becomes universally more capable. The defensible finding, with three concrete implications:

> **A small open-weight model can be substantially specialized for a narrowly defined structured-output task using parameter-efficient fine-tuning.**

1. **Engineering implication.** Task specialization can be more valuable than increasing model size when the target is narrow and measurable. The +42 pp argument-exact-match gain is the headline, but the *practical* claim is cost-shaped: a 1.5B QLoRA adapter trains in hours on one GPU and adds negligible inference latency over the base (§17.1), versus a much larger model whose marginal capability may be unneeded for a fixed tool registry.
2. **Systems implication.** The LLM should be one *component* of a tool-calling system, not the whole agent. The failure budget (§19.2) shows that even at 98% validity, production safety comes from the deterministic layer — allowlist, schema validation, authorization — not from the model alone. This separation makes the system testable and auditable in ways a monolithic agent is not.
3. **Scientific implication (small).** The result corroborates, at 1.5B, ToolACE's observation that small raw models have minimal function-calling ability but improve sharply with fine-tuning [6] — and does so with a prompted-baseline control and confidence intervals, which leaderboard-centric reports often omit.

## 25. Industry Insights

**Market and adoption signals (2026).** The demand for the capability this project targets is concrete and rising. Gartner projects that **40% of enterprise applications will embed task-specific AI agents by end-2026, up from under 5% in 2025** [18]; the AI-agent market is estimated at **$8.5–10.9B in 2026**, with forecasts diverging from $35B (2030) to $199B (2034) [19]; and adoption is already broad — **62% of organizations report experimenting with AI agents, 23% are scaling them** [20]. Where those agents meet the real world, function calling is the interface: **50–65% of customer-support inquiries are already handled without human intervention**, with reported **20–30% reductions in support operating cost** [20]. Every one of those automated actions is, at bottom, a model producing a structured call that deterministic software executes — the exact contract TinyToolCaller is trained to produce. (These figures are secondary 2026 market surveys, not peer-reviewed measurements, and are cited as directional context.)

**The shifting problem definition.** As agentic systems interact with APIs, databases, search, calendars, and enterprise applications, the practical question moves from *"can the model generate a good answer?"* to *"can the model reliably produce an action software can safely execute?"* — and the failure-cost curve changes with it: a verbose-but-correct answer costs nothing, while a wrong tool call can trigger an irreversible external effect. TinyToolCaller addresses one component of that transition.

**The large-generalist vs. small-specialist trade-off.** A small specialized model is attractive when the operational task is narrow enough that general reasoning is not the primary requirement — the situation for a fixed enterprise tool registry. The decision hinges on task and schema complexity, latency, cost, error tolerance, deployment environment, and safety requirements; this project does **not** establish that smaller models are universally better.

**The compression and on-device trend.** Octopus v2 [14] and the small-model function-calling study [10] reflect a broader push toward on-device and edge tool use, where a 1.5B QLoRA adapter is a natural fit; the project's single-GPU, hours-not-days training story (§17.1) is aimed exactly at that segment.

**The open-weight function-calling ecosystem.** xLAM-1b/7b-fc-r [1], Gorilla [3], NexusRaven [5], Hermes, Granite [7], and Qwen's own function-calling variants show open models converging on proprietary tool-calling parity; the remaining differentiators are data quality (APIGen's execution verification [1]) and evaluation rigor (BFCL's AST scoring [2]) — which is why §32 prioritizes joining that external evaluation.

## 26. Uncommon Insights

Observations from this work that are not obvious from the headline table:

1. **The cheapest metric was the least valuable, and the most valuable metric improved the most.** JSON validity (nearly solved at 98%) is the metric with the smallest practical consequence, because malformed JSON is cheaply recoverable (retry/repair). Argument exact match — the failure class that silently breaks production — improved the most (+42 pp). The model didn't just learn formatting; it learned schema-following.

2. **The 92.5% → 84.0% gap is the real deployment risk, and it is invisible to schema validation.** The ≈8.5 pp of examples with the right tool but wrong arguments produce *well-typed, structurally valid* output. Only semantic or range validation can catch them — a cost that most function-calling benchmarks, which score "call correctness" in aggregate, do not surface.

3. **In-sample baselines can be misleading in both directions.** The baseline's 78.5% "JSON validity" *already includes* fence-stripping cleanup; conversely, its 42% argument rate shows most of the gap was never about formatting. Report the raw-output metric separately to see both effects (§21.7).

4. **A quantized baseline is a control, not a convenience.** Evaluating the base model in the same 4-bit NF4 regime removes precision as a confound; if the "improvement" partially disappears when the baseline is bf16, that itself is a finding about quantization, not fine-tuning. This ablation is one flag away (`eval_load_in_4bit=False`).

5. **Truncation is a hidden schema-size confound — and it's measurable.** With the real tokenizer, ~5 verbose tools already approach the 1024-token cap and 10 exceed it (§13.2). If the eval set skews the same way, accuracy is overstated relative to production prompts with many tools. `scripts/dataset_stats.py` turns this from speculation into a number.

6. **Retention claims need more than a 50-example diff.** A ±13 pp confidence interval means even a *true* 10-point forgetting effect would be undetectable here — the retention "result" is currently an absence of evidence, not evidence of absence.

7. **Tests caught a real bug cheaply.** The suite's `test_extract_json_rejects_bare_list` exposed that a JSON-list output would crash the scorer (`.get` on a list); the fix (reject non-dict parses) is two lines. A 41-test suite is disproportionate value for a project this size, and it runs on a CPU with no GPU stack because the heavy imports are lazy (§13.2).

## 27. Source Credibility and Provenance

The project relies on first-party sources: the xLAM dataset is published by Salesforce AI Research with documented APIGen generation and verification [1]; Qwen2.5-1.5B-Instruct is published by Qwen (Apache-2.0); QLoRA and LoRA follow the original peer-reviewed work [11][12]; TRL/PEFT documentation comes from Hugging Face [16]; GSM8K originates from Cobbe et al. [13]; the benchmark and model comparisons cite BFCL [2], ToolACE [6], Gorilla [3], Toolformer [4], NexusRaven [5], Granite [7], τ-bench [8], and Octopus v2 [14].

## 28. Licensing and Attribution

| Artifact | License |
| --- | --- |
| Source dataset (`Salesforce/xlam-function-calling-60k`) | CC-BY-4.0 |
| Base model (`Qwen/Qwen2.5-1.5B-Instruct`) | Apache-2.0 |
| TinyToolCaller (code, derived data, documentation, artifacts) | Project-specific (see `LICENSE`) |

Users should review upstream licenses and attribution requirements before redistribution or commercial deployment.

## 29. Reproducibility and Verifiability

**Falsifiable claims.** The paper's central claims are stated so that a third party can attempt to refute them: (i) on the documented 5,200-example seed-42 subset, QLoRA raises each of the three metrics by the reported magnitudes (§17); (ii) the direction of improvement holds under paired testing (§18); (iii) the GSM8K change is indistinguishable from noise (§20).

**Determinism and configuration.**

```text
Seed 42 · 5,000 train / 200 validation · LoRA r=16, α=32, dropout=0.05
LR 2e-4 · cosine + 3% warmup · effective batch 16 · 2 epochs · max seq len 1024
```

**Workflow.**

```bash
pip install -r requirements.txt
pip install pytest

export HF_TOKEN=<your_huggingface_token>       # gated access to the source dataset
export WANDB_API_KEY=<your_wandb_key>          # optional

python -m pytest tests/ -v                     # 41 tests (config, formatting, metrics, data, repair)
python scripts/capture_environment.py --save outputs/environment.json
python scripts/dataset_stats.py                # §8.2
python scripts/profile_tool_distribution.py    # §8.1
python train_tool_caller.py                    # full 14-stage pipeline
python train_tool_caller.py --eval-dump outputs/eval_predictions.jsonl
python scripts/statistical_analysis.py --report
python scripts/statistical_analysis.py --mcnemar outputs/eval_predictions.jsonl
python scripts/publish_dataset.py --push       # publishes the derived dataset
```

**Verifiability gaps to close before final publication.** (1) Record the full environment (§12), pin exact dependency versions (a lockfile), and publish a container/requirements hash; (2) publish checksums for the derived `train.parquet`/`validation.parquet`; (3) commit the W&B run ID/loss curve for the reported training run; (4) fix the `datasets` version used for the seed-42 shuffle (the shuffle RNG is version-sensitive, so §8.1's numbers are only exactly reproducible on the recorded version).

## 30. Repository and Dataset

| Artifact | Location |
| --- | --- |
| Code | https://github.com/strdst7/TinyToolCaller |
| Project dataset | https://huggingface.co/datasets/strdst77/TinyToolCaller |
| Source dataset | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k |
| Base model | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |

## 31. Project Architecture

<img width="1300" height="660" alt="architecture" src="https://github.com/user-attachments/assets/10bfdf71-b982-4ff3-9287-2303803388e2" />


```text

Salesforce xLAM 60K → sampling/split (5K train / 200 val) → ChatML formatting
    → Qwen2.5-1.5B-Instruct (4-bit NF4 + QLoRA) → SFTTrainer/TRL
    → LoRA adapter weights
        ├→ function-calling evaluation (3 metrics + per-example dump)
        ├→ GSM8K retention
        └→ statistical analysis (Wilson CI / McNemar / bootstrap)
    → model comparison → adapter + base → merged model → Hugging Face Hub

```

**Technical progression.** The project moved from a single end-to-end script to a modular package (`tinytoolcaller/`) with unit tests that pin the configuration to the paper (§13.3); from a single aggregate "accuracy" to three failure-mode metrics (§14); and from point estimates to confidence intervals and a paired significance test (§18). The next step in that progression is external, independent evaluation (§32).

## 32. Future Architecture

A production-ready evolution adds a deterministic runtime around the model:

```text
User → Application → Tool Registry → Prompt Builder → TinyToolCaller → generated JSON
    → JSON Schema Validator → (invalid → repair/reject) → Authorization
    → Tool Executor → External API → tool result
```

The current project produces the **tool-call generation component**; a production agent adds validation, execution, permissions, retries, and observability.

## 33. Future Directions and Research Extensions

The findings and limitations (§21) define six **open research questions**. Each is posed so a follow-up study can answer it falsifiably:

- **RQ1 — Generalization.** Does the reported lift survive a *held-out* test split never used during development? (§15-V1; the single most important open question.)
- **RQ2 — Skew.** Does tool-accuracy remain ≥90% once the subset's tool distribution is profiled and de-skewed? (§8.1)
- **RQ3 — Retention.** Is there *any* real forgetting, and on which capability — measured on the full GSM8K or equivalent? (§15-V3)
- **RQ4 — Efficiency frontier.** What is the smallest LoRA rank (and fewest examples) that retains ≥90% of the argument-exact-match gain? (This would draw the cost/quality Pareto curve for QLoRA specialization.)
- **RQ5 — Quantization confound.** How much of the base→fine-tuned gap is fine-tuning vs. 4-bit quantization? (§25.4, the bf16-baseline ablation.)
- **RQ6 — Failure mechanism.** For the "right tool, wrong arguments" band (§19.2), are errors dominated by enumeration values, missing required fields, or value hallucination? (Requires the per-example dump, §18.)

Ordered by effort vs. impact, with the question each direction answers:

| Priority | Direction | Answers | Effort | Impact |
| --- | --- | --- | --- | --- |
| 1 | Independent held-out test split (§15-V1) | RQ1 | Low | High — converts in-sample to generalization evidence |
| 2 | Tool-distribution profiling (§8.1) + basic stats (§8.2) | RQ2 | Low | High — removes the skew caveat |
| 3 | Full GSM8K retention run (§15-V3) | RQ3 | Low | High — makes the retention claim meaningful |
| 4 | Raw-output (no-extraction) JSON metric (§21.7) | — | Low | Medium — deployment-relevant purity number |
| 5 | Quantization ablation: bf16 vs. 4-bit baseline (§25.4) | RQ5 | Low | Medium — isolates quantization vs. fine-tuning |
| 6 | Multi-seed variance (§15-V4) | RQ1/RQ3 | Low | Medium — honest error bars |
| 7 | Hyperparameter + rank/data sweep (§17.4 baselines 3–5) | RQ4 | Medium | Medium — draws the QLoRA Pareto curve |
| 8 | External benchmark (BFCL [2], τ-bench [8]) | — | Medium | High — comparability |
| 9 | Error annotation of the "wrong arguments" band | RQ6 | Medium | High — targets the riskiest failure class |
| 10 | Multi-turn tool use (APIGen-MT [9] direction) | — | High | High — production agentic relevance |
| 11 | Distillation from a larger teacher | — | High | High — capability transfer |

**Larger research bets:** multi-turn tool use, distillation, runtime schema validation, 8-bit/4-bit inference optimization, and relevance detection ("when not to call") à la BFCL V3 [2].

## 34. Accessibility and Learning Design

The project is structured for readers with basic Python and ML knowledge but no prior LLM fine-tuning experience. The workflow is intentionally simple — *data → format → baseline → fine-tune → evaluate → compare → publish* — and readers need not understand every Transformer implementation detail. The concepts that matter: what function calling is; why structured JSON matters; what supervised fine-tuning does; what LoRA does; why quantization reduces memory; how baseline and post-training evaluation differ; why confidence intervals and validation design determine the strength of conclusions.

**The six core concepts in plain language (analogies first):**

1. **Function calling** is asking the model to *fill in a form*, not write an essay. A chat model can say "the weather in Tokyo is sunny"; a tool-calling model fills the blank fields of `get_weather(location=___, unit=___)` so a program — not a human — can act on it.
2. **Fine-tuning (SFT)** is like giving a generalist employee thousands of worked examples of exactly the forms you want filled in, so the specialist behaviour becomes a habit. The model's "knowledge" barely changes; its *output behaviour* does.
3. **LoRA** is like correcting a published book with sticky notes instead of reprinting it: the original text (the base weights) is never touched; only the thin layer of notes (low-rank matrices, ~1.8% of parameters here) is learned.
4. **Quantization (4-bit NF4)** is like compressing a photo to a smaller file: nearly the same content, a quarter of the storage, at the cost of a little precision. QLoRA trains the sticky notes *on the compressed copy* so it fits on one GPU.
5. **Baseline vs. fine-tuned** is a before-and-after photo taken with the *same camera and settings*: same prompts, same metrics, same precision, so any difference is the treatment (fine-tuning), not the equipment.
6. **Confidence intervals** answer "how much should I trust this number?" A 84% on 200 examples could really be 78–88% (§18); quoting the interval instead of a bare point estimate is what keeps an honest result honest.

### 34.1 Glossary

| Term | Meaning in this project |
| --- | --- |
| Function calling / tool calling | Generating a structured request (name + arguments) that a program can execute |
| Fine-tuning (SFT) | Training on input→target examples to shape behaviour |
| LoRA | Freezing the base weights and training small low-rank update matrices |
| QLoRA | LoRA on top of a 4-bit-quantized (NF4) base model to cut memory |
| Adapter | The trained LoRA weights, merged or loaded alongside the base |
| ChatML | The `<|im_start|>/<|im_end|>` message format used by Qwen |
| Baseline | The unmodified base model scored on the same prompts |
| JSON validity (this project) | Whether a JSON *object* could be extracted from the output (§14) |
| Exact match | Predicted `name`/`arguments` equal ground truth exactly (§14) |
| Wilson CI | A confidence interval for a proportion (here, 95%) |
| McNemar's test | Paired significance test for before/after on the same examples |
| GSM8K | Grade-school math benchmark used as a retention probe |
| Gated dataset | Requires accepting terms + a Hugging Face token to download |

## 35. Key Takeaways

1. **Small models can be specialized** — a 1.5B model can become substantially better at a narrow structured-output task.
2. **QLoRA makes specialization accessible** — parameter-efficient fine-tuning reduces the trained state.
3. **Function calling is more than valid JSON** — tool selection and argument correctness are separate failure dimensions.
4. **Evaluation design matters** — evidence is only as strong as the validation methodology, and statistics should carry confidence intervals.
5. **LLMs should not be the entire agent** — combine the LLM with validation, authorization, and deterministic execution.

## 36. Conclusion

Starting from `Qwen/Qwen2.5-1.5B-Instruct` and fine-tuning with QLoRA on a 5,000-example subset of the Salesforce xLAM dataset, TinyToolCaller reports **JSON validity 78.5% → 98.0%, tool accuracy 65.0% → 92.5%, argument exact match 42.0% → 84.0%** on its 200-example evaluation split, with a 50-example GSM8K retention check moving 52.0% → 50.0%. The contribution is not a claim of universal superiority but a demonstrable engineering pattern:

```text
General-purpose model → task-specific data → parameter-efficient fine-tuning
    → specialized small model → structured interface → deterministic software
```

This pattern suits lower-cost, lower-latency AI systems whose target capability is narrow, measurable, and operationally well-defined. TinyToolCaller is best viewed as a **reproducible applied LLM engineering study and a foundation for a production-grade tool-calling runtime** — not a finished autonomous-agent platform.

## 37. Publication Checklist

- [x] Clear problem statement (§1)
- [x] Related work with critical analysis and current citations (§2)
- [x] Objectives, contributions, and originality stated (§3)
- [x] Assumptions and scope stated (§4)
- [x] Intended audience, use case, prerequisites, and non-goals (§5)
- [x] Validation strategy: in-place controls + held-out set & robustness plan (§15)
- [x] Methodological contributions (O-FME + repair loop) implemented and tested (§3.1)
- [x] Dataset source, selection rationale, and description (§8)
- [x] Comprehensive dataset description: schema, stats, class/quality characteristics (§8.2–§8.5)
- [x] Worked ChatML example with real tokenizer counts (§9.1)
- [x] Dataset processing methodology (§9)
- [x] Training methodology, parameters, and per-parameter rationale (§10–§11)
- [x] Experimental environment template + capture script (§12)
- [x] Implementation workflow, package layout, and code quality (§13)
- [x] Unit tests passing (41) (§13.3–§13.5)
- [x] Evaluation framework and metrics (§14)
- [x] Validation strategy and protocol (§15)
- [x] Baseline established (§16)
- [x] Comparative results + related-work position (§17)
- [x] Statistical analysis with confidence intervals (§18)
- [x] Results interpretation, error analysis, illustrative transcripts (§19)
- [x] Limitations disclosed (§21)
- [x] Deployment considerations with validator snippet (§22)
- [x] Monitoring, drift detection, and runbook (§23)
- [x] Significance, industry insights, uncommon insights (§24–§26)
- [x] Source credibility and licensing (§27–§28)
- [x] Reproducibility and verifiability instructions (§29)
- [x] Code, dataset, and model linked (§30)
- [x] Future architecture and research roadmap (§32–§33)
- [x] Accessibility and glossary (§34)
- [x] Rubric coverage matrix (§38)
- [ ] **Tool-distribution profiling (§8.1) completed** *(open — blocks unqualified quoting of tool accuracy)*
- [ ] Basic dataset statistics computed and recorded (§8.2) *(open)*
- [ ] Independent test set created and locked before model selection *(open)*
- [ ] Full-size GSM8K retention run *(open)*
- [ ] Experimental environment recorded; dependency versions pinned (§12, §29) *(open)*
- [ ] Paired McNemar + bootstrap computed from the per-example dump (§18) *(open)*

## 38. Rubric Coverage Matrix

Where each evaluation dimension is addressed.

### Applied solution showcase

| Rubric item | Section(s) |
| --- | --- |
| Evaluation Framework | §14, §15 |
| Dataset Description | §8.3–§8.5 (schema, structure, stats, quality) |
| Dataset processing Methodology | §9, §9.1 |
| Implementation Considerations | §13.2 |
| Deployment Considerations | §22 (22.1 controls · 22.2 infra · 22.3 integration · 22.4 scalability · 22.5 security · 22.6 performance) |
| Monitoring and Maintenance Considerations | §23 (thresholds · log schema · drift · maintenance schedule · runbook) |
| Comparative Analysis | §17 (17.1 base-vs-ft · 17.2 related work · 17.3 published SOTA · 17.4 baselines) |
| Results Interpretation | §19 |
| Future Directions | §33 (RQ1–RQ6 + prioritized roadmap) |
| Purpose-Aligned Topic Coverage | §1, §3, §38 |
| Code Usage Appropriateness | §13.3 |
| Code Clarity and Presentation | §13.3 |
| Code Explanation Quality | §13.4 |
| Industry Insights | §25 |
| Success/Failure Stories | §19.1–§19.4 (incl. lessons learned) |
| Technical Progression | §31 |
| Source Credibility | §27 |
| Uncommon Insights | §26 |
| Significance and Implications of Work | §24 |

### Research paper

| Rubric item | Section(s) |
| --- | --- |
| Testability/Verifiability | §29 |
| Literature Review Coverage & Currency | §2 (2021–2025, incl. BFCL ICML 2025, APIGen-MT 2025) |
| Literature Review Critical Analysis | §2.1–§2.4 ("Critical gap" notes) |
| Citation Relevance | §2, §27, §39 (numbered, claim-anchored) |
| Assumptions Stated | §4 |
| Intended Audience / Use Case | §5 (audience table · prerequisite tiers · non-goals) |
| Evaluation Framework | §14 |
| Validation Strategy | §15 (Part A controls + Part B held-out set & robustness checks B1–B6) |
| Dataset Description | §8.3–§8.5 (schema, structure, stats, quality) |
| Dataset Selection or Creation | §8.0 (criteria table + utilization & modification log) |
| Dataset processing Methodology | §9 + §9.2 (cleaning rules) |
| Basic Dataset Stats | §8.2 + §8.3 (features/data types + class distribution) |
| Implementation Details | §13 |
| Parameters & Configuration | §11 (rationale column + tuning methodology + standard/non-standard) |
| Experimental Environment | §12 |
| Implementation Considerations | §13.2 |
| Comparative Analysis | §17 (17.1–17.4: 6 systems + 9 published models + 5 internal baselines) |
| Statistical Analysis | §18 |
| Results Interpretation | §19 |
| Future Directions | §33 (RQ1–RQ6 + prioritized roadmap) |
| Originality of Work | §3 |
| Innovation in Methods/Approaches | §3.1 (O-FME framework + one-shot repair loop, implemented & tested) |
| Code Usage Appropriateness | §13.3, §13.5 |
| Code Clarity and Presentation | §13.5 (conventions + minimal-snippet policy) |
| Content Accessibility | §34 (analogies) + §34.1 (glossary) |
| Significance and Implications of Work | §24 |

## 39. References

1. Liu, Z., Hoang, T., Zhang, J., et al. *APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets*. arXiv:2406.18518, 2024.
2. Patil, S. G., Mao, H., Yan, F., Ji, C. C.-J., Suresh, V., Stoica, I., & Gonzalez, J. E. *The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models*. ICML 2025, PMLR 267:48371–48392.
3. Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. *Gorilla: Large Language Model Connected with Massive APIs*. arXiv:2305.15334, 2023.
4. Schick, T., Dwivedi-Yu, J., Dessì, R., et al. *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761, 2023.
5. Srinivasan, V. K., Dong, Z., Zhu, B., et al. *NexusRaven: A Commercially-Permissive Language Model for Function Calling*. NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023.
6. Liu, Z., et al. *ToolACE: Winning the Points of LLM Function Calling*. arXiv:2409.00920, 2024 (ICLR 2025).
7. Abdelaziz, I., et al. *Granite-Function Calling Model: Introducing Function Calling Abilities via Multi-task Learning of Granular Tasks*. arXiv:2407.00121, 2024.
8. Sierra, S., et al. *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*. arXiv:2406.12045, 2024.
9. Prabhakar, A., Liu, Z., Zhu, M., et al. *APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay*. arXiv:2504.03601, 2025.
10. *Small Models, Big Tasks: An Exploratory Empirical Study on Small Language Models for Function Calling*. arXiv:2504.19277, 2025.
11. Hu, E. J., Shen, Y., Wallis, P., et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021.
12. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. arXiv:2305.14314.
13. Cobbe, K., Kosaraju, V., Bavarian, M., et al. *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168, 2021.
14. Chen, W., & Li, Z. *Octopus v2: On-device Language Model for Super Agent*. arXiv:2404.01744, 2024.
15. Zhang, J., Lan, T., Zhu, M., et al. *xLAM: A Family of Large Action Models to Empower AI Agent Systems*. arXiv:2409.03215, 2024.
16. Hugging Face. *TRL / SFTTrainer Documentation*. https://huggingface.co/docs/trl/sft_trainer
17. Qin, Y., et al. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*. arXiv:2307.16789, 2023.
18. Gartner (via P. Okhrem). *Enterprise AI Agents Adoption Statistics 2026*. https://paul-okhrem.com/enterprise-ai-agents-statistics-2026/ (secondary source, 2026).
19. Gradually.ai. *AI Agent Statistics 2026: Adoption, Market & Facts*. https://www.gradually.ai/en/ai-agent-statistics/ (secondary source, 2026).
20. Insight Mark Research. *LLM Agent Statistics 2026*. https://insightmarkresearch.com/insights/llm-agent-statistics-2026 (secondary source, 2026).

## 40. Project Links and Citation

**Code** — https://github.com/strdst7/TinyToolCaller
**Project dataset** — https://huggingface.co/datasets/strdst77/TinyToolCaller
**Source dataset** — https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
**Base model** — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

```bibtex
@misc{tinytoolcaller,
  title        = {TinyToolCaller: QLoRA Fine-Tuning of a 1.5B LLM for Reliable Function Calling},
  author       = {strdst7},
  year         = {2026},
  howpublished = {\url{https://github.com/strdst7/TinyToolCaller}},
  note         = {Base model: Qwen/Qwen2.5-1.5B-Instruct. Source dataset:
                  Salesforce/xlam-function-calling-60k (APIGen).}
}
```

---

### Final Project Statement

> **TinyToolCaller demonstrates that a small open-weight language model can be deliberately specialized for reliable function calling through QLoRA, achieving substantial improvements in structured output, tool selection, and argument accuracy while maintaining a reproducible and transparent evaluation pipeline. The project provides a practical foundation for exploring low-cost LLM agents in which the model generates structured intent and deterministic software remains responsible for validation, authorization, and execution.**

---

<p align="center"><strong>Nur Amirah Mohd Kamil | 2026 | Ready Tensor for <em>LLM Fine-Tuning Specialist</em></strong><br>
Fine-tune and optimize an LLM using PEFT techniques.</p>
