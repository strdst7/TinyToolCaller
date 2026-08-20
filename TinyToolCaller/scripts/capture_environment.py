#!/usr/bin/env python3
"""Capture and print the runtime environment for reproducibility (§12, §28).

Records the GPU, CUDA/PyTorch, and library versions that must be pinned for a
third party to reproduce the training run. Prints a Markdown table and, with
``--save``, writes ``environment.json`` next to the run outputs.

Usage:
    python scripts/capture_environment.py
    python scripts/capture_environment.py --save outputs/environment.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys


def collect() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    # Best-effort: heavy deps may be absent in CI environments.
    for name, get_ver in [
        ("torch", lambda m: m.__version__),
        ("transformers", lambda m: m.__version__),
        ("trl", lambda m: m.__version__),
        ("peft", lambda m: m.__version__),
        ("datasets", lambda m: m.__version__),
        ("bitsandbytes", lambda m: m.__version__),
        ("wandb", lambda m: m.__version__),
        ("huggingface_hub", lambda m: m.__version__),
    ]:
        try:
            module = __import__(name)
            env[name] = get_ver(module)
        except Exception:
            env[name] = None
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1
            )
    except Exception:
        env["cuda_available"] = None
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", default=None,
                        help="Write the report as JSON to this path.")
    args = parser.parse_args()

    env = collect()
    print("| Item | Value |")
    print("| --- | --- |")
    print(f"| Python | {env.get('python')} |")
    print(f"| Platform | {env.get('platform')} |")
    print(f"| GPU | {env.get('gpu_name', 'TBD')} "
          f"({env.get('gpu_total_gb', '?')} GB)" if env.get("gpu_name")
          else "| GPU | TBD |")
    for key in ("torch", "transformers", "trl", "peft", "datasets",
                "bitsandbytes", "wandb", "huggingface_hub"):
        print(f"| {key} | {env.get(key) or 'TBD'} |")
    print(f"| CUDA available / version | {env.get('cuda_available')} / "
          f"{env.get('cuda_version') or 'TBD'} |")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(env, fh, indent=2)
        print(f"\nWrote {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
