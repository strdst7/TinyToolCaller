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




## What is it?

TinyToolCaller is a **small AI model** (1.5 billion parameters, built from Qwen2.5) that has been specially trained to do **one job very well**:

> When someone asks for something in normal language, it responds with a clean, machine-readable instruction (a "tool call") that a computer program can act on.

**Example**

| A normal chatbot says… | TinyToolCaller outputs… |
| --- | --- |
| *"The weather in Tokyo is likely to be sunny."* | `{"name": "get_weather", "arguments": {"location": "Tokyo"}}` |

No extra chatter, no explanations, no formatting errors — just the instruction.

## Why this matters

Most AI assistants reply in full sentences. That is great for chatting, but **useless when software needs to act** — book a meeting, check an order status, run a calculation. Software needs a strict, predictable format it can validate and execute. TinyToolCaller turns human language into exactly that format.

## How it was built (3 steps)

1. **Started small** — took an existing open-source model, `Qwen2.5-1.5B-Instruct`.
2. **Taught it the task** — trained it on **5,000 examples** of *"request + list of available tools → correct tool call"*, taken from a public dataset (Salesforce's xLAM dataset, 60,000 examples).
3. **Used a cheap training method** — **QLoRA** (a memory-efficient fine-tuning technique) so the whole thing runs on a **single GPU** in hours, not days, without a supercomputer.

## The results

Before training, the model filled in the tool call *exactly correctly* only **42%** of the time. After training: **84%**.

| Skill | Before | After | Plain meaning |
| --- | ---: | ---: | --- |
| Produce valid JSON | 78.5% | **98.0%** | Almost always outputs clean, parseable JSON |
| Pick the right tool | 65.0% | **92.5%** | Almost always chooses the correct function |
| Fill in the arguments correctly | 42.0% | **84.0%** | The big one — the values are right too |
| General math ability (unchanged) | 52.0% | **50.0%** | No meaningful loss of general skill |



<img width="1300" height="660" alt="architecture" src="https://github.com/user-attachments/assets/bc5b252d-d2cc-4715-99f1-761fc1151590" />



## Honest caveats (read before sharing the numbers)

- The results come from the **same 200 examples used during development** — they clearly show the improvement, but they are not a guaranteed score on brand-new data. A separate held-out test set is the project's next step.
- This project does **not** claim to beat GPT-4 or other big models. Its point is narrower: **a small, cheap model can be made highly reliable for one specific, well-defined job.**
- The model produces the instruction; **software** (validation, permissions, execution) must still own the action — the model is one component of a system, not the whole system.

## Links

- **Code** — https://github.com/strdst7/TinyToolCaller
- **Project dataset** — https://huggingface.co/datasets/strdst77/TinyToolCaller
- **Source dataset** — https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- **Base model** — https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

---

*Nur Amirah Mohd Kamil · 2026 · Ready Tensor — LLM Fine-Tuning Specialist*

