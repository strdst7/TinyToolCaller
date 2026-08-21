"""One-shot repair loop for malformed tool calls (publication §3.1, §22.1).

Method contribution: a minimal retry-with-repair that re-prompts the model
with its own offending output plus a compact instruction, then re-extracts.
The loop is injectable (``generate_fn``), so it is testable without torch and
reusable with any decoder.

Design decisions (justified in §3.1):
  * ONE repair attempt by default — each attempt costs latency and tokens, and
    the marginal recovery drops sharply after the first retry;
  * the model sees its OWN output, not a generic error — the failure signature
    is the most informative signal;
  * ``extract_json`` is the same parser used at evaluation time (§14), so the
    repair loop and the reported metrics share one definition of "valid".
"""

from __future__ import annotations

from .formatting import extract_json

REPAIR_INSTRUCTION = (
    "Your previous response was not valid JSON. Respond with ONLY a JSON "
    "object containing \"name\" and \"arguments\", with no markdown, no "
    "explanations, and no other text.\n\nPrevious response:\n"
)


def repair(raw: str, generate_fn, prompt: str, max_attempts: int = 1):
    """Return (raw_text, attempts) after up to ``max_attempts`` repairs.

    ``generate_fn`` is ``callable(prompt) -> raw_text``; injecting it keeps
    this function pure of model/tokenizer state and unit-testable.
    """
    attempts = 0
    current = raw
    while extract_json(current) is None and attempts < max_attempts:
        current = generate_fn(prompt + "\n" + REPAIR_INSTRUCTION + current)
        attempts += 1
    return current, attempts
