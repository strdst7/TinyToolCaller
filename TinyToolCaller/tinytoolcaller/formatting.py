"""Pure prompt-construction and extraction helpers (no heavy dependencies).

These functions implement the documented output contract and evaluation
parsing (publication §9 and §14). They are import-safe in CPU/CI environments
so they can be unit-tested without torch/trl/peft.
"""

from __future__ import annotations

import json
import re

from .config import SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# ChatML construction (publication §9)
# --------------------------------------------------------------------------- #
def build_messages(example: dict) -> list[dict]:
    """System -> user (available tools + request). Assistant turn added later.

    The user message layout matches the publication §9:

        Available Tools:
        <JSON tool schemas>

        User Request:
        <natural-language query>
    """
    tools_json = json.dumps(example["tools"], ensure_ascii=False)
    user = f"Available Tools:\n{tools_json}\n\nUser Request:\n{example['query']}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ground_truth(example: dict) -> dict:
    """Return the single expected call: the first entry of ``answers``.

    xLAM examples use ``answers`` (a list); some variants use a bare dict.
    Both shapes are tolerated.
    """
    answers = example.get("answers", example.get("answer"))
    if isinstance(answers, str):
        answers = json.loads(answers)
    if isinstance(answers, list):
        return answers[0]
    return answers


def format_for_training(example: dict, tokenizer) -> dict:
    """ChatML-formatted supervised example -> a single ``text`` field."""
    messages = build_messages(example)
    messages.append({
        "role": "assistant",
        "content": json.dumps(ground_truth(example), ensure_ascii=False),
    })
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def format_for_inference(example: dict, tokenizer) -> str:
    """ChatML prompt without the assistant turn (for generation/evaluation)."""
    return tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )


# --------------------------------------------------------------------------- #
# Evaluation parsing (publication §14 / §21.7)
# --------------------------------------------------------------------------- #
def extract_json(text: str):
    """Extract a JSON object from raw model output.

    The documented pipeline strips ``` ```json ``` fences and surrounding prose
    before attempting ``json.loads`` — so the JSON-validity metric measures
    *extractable* JSON, not raw-output purity (publication §21.7).
    """
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan of the first '{...}' region, tolerant of wrappers.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def extract_gsm8k_answer(text: str) -> str:
    """Final numeric answer for GSM8K. Shared by both models for a fair A/B."""
    final = re.findall(r"####\s*(-?[\d,]+\.?\d*)", text)
    if final:
        return final[-1].replace(",", "")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else ""


def normalise_number(value: str) -> str:
    """Normalise a numeric string for GSM8K comparison."""
    value = value.replace(",", "").strip()
    if value.endswith("."):
        value = value[:-1]
    try:
        return f"{float(value):g}"
    except ValueError:
        return value
