# TinyToolCaller — Production Deployment Infrastructure

## What This Is

Production deployment infrastructure for TinyToolCaller, a QLoRA fine-tuned 1.5B function-calling model. Takes the existing training pipeline and publication (§22 deployment design) and implements the full production stack: FastAPI inference server with the validation safety layer, Docker packaging, CI/CD automation, and optional Kubernetes orchestration.

## Core Value

Deploy the TinyToolCaller model behind a production-grade REST API with deterministic safety controls (JSON validation → allowlist → schema check → authorization) so that any application can call it as a structured tool-calling component without building infrastructure from scratch.

## Requirements

### Validated

- ✓ Publication §22 deployment design exists (runtime controls, infrastructure specs, integration contract, scalability, security, performance targets) — existing
- ✓ Publication §23 monitoring design exists (alert thresholds, log schema, drift detection, maintenance schedule) — existing
- ✓ 41-test suite for pure functions (formatting, config, metrics) — existing
- ✓ GitHub Actions CI workflow (pytest + flake8 on 3.9-3.11) — existing
- ✓ LoRA adapter + merged model published on Hugging Face Hub — existing
- ✓ `model.py` with quantized model loading and LoRA attachment — existing
- ✓ `repair.py` with dependency-injected JSON repair loop — existing
- ✓ `formatting.extract_json` — shared parser between evaluation and deployment — existing

### Active

- [ ] **INFRA-01**: FastAPI inference server exposing `/tool-call` endpoint
- [ ] **INFRA-02**: Validation stack implementation (extract_json → allowlist → jsonschema → authorization)
- [ ] **INFRA-03**: CLI inference mode for local/script usage
- [ ] **INFRA-04**: Dockerfile with multi-stage build (CPU-only and GPU variants)
- [ ] **INFRA-05#### docker-compose** for local deployment
- [ ] **INFRA-06**: CI/CD pipeline (GitHub Actions): test → build → publish Docker image → deploy
- [ ] **INFRA-07#### Helm chart or Kubernetes manifests for cloud deployment
- [ ] **INFRA-08**: Health check and readiness probe endpoints
- [ ] **INFRA-09**: Configuration via environment variables (model path, allowlist, log level)
- [ ] **INFRA-10**: Request logging with the §23 log schema
- [ ] **INFRA-11**: Batch inference mode for offline/async processing
- [ ] **INFRA-12**: Pin dependency versions in requirements.txt and generate lockfile
- [ ] **INFRA-13**: Clean up orphaned `TinyToolCaller-files/` directory
- [ ] **INFRA-14**: Update README with deployment guide and architecture docs

### Out of Scope

- Multi-turn tool orchestration — out of scope per §4; deployment supports single-call contract
- vLLM / TGI integration — deferred; the FastAPI server serves as the reference implementation
- Authentication layer (API keys, OAuth) — deployment provides the auth hook; specific auth is deployment-specific
- Horizontal autoscaling — Kubernetes HPA is documented but not implemented; single-GPU serving is the target
- Model monitoring dashboard — log schema is defined; actual dashboard is deployment-specific

## Context

The codebase is mature for research but has zero deployment code. The publication already specifies the integration contract (`POST /tool-call`), validation stack (6 layers), log schema, and alert thresholds — this project implements all of that as working code.

The model is 1.5B → 4-bit fits in ~1 GB VRAM, merged bf16 ~3 GB. Target hardware is single GPU (RTX 4090, A10G, L4) or CPU for development. Inference latency target is p95 < 500 ms.

The repo has an orphaned `TinyToolCaller-files/` directory containing test files and data-quality functions that should be merged into the active codebase. This cleanup is included in Phase 1.

## Constraints

- **Tech stack**: Python 3.10+, FastAPI, PyTorch + Transformers, Docker, GitHub Actions
- **Hardware**: Single NVIDIA GPU (recommended: RTX 4090 24 GB or cloud A10G/L4); CPU fallback for dev
- **Model format**: Hugging Face `transformers`-compatible (merged model or base + LoRA adapter)
- **Integration**: Must match the §22.1 validation stack (6-layer safety gate)
- **License**: Apache-2.0 (inherited from existing code)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| FastAPI for inference server | Async by default, automatic OpenAPI docs, Pydantic validation, widely adopted | ✓ Good |
| Multi-stage Docker build | GPU image with CUDA deps vs CPU-only dev image from same Dockerfile | ✓ Good |
| GitHub Actions for CI/CD | Already used for CI; minimal new infra | ✓ Good |
| Environment variable config | 12-factor app pattern; no config files in containers | ✓ Good |
| Single-GPU target | Model fits comfortably on one consumer GPU; no multi-GPU complexity needed | ✓ Good |
| LoRA adapter hot-swap not in v1 | Adds complexity; single-adapter serving for initial release | — Pending |
| vLLM/TGI integration deferred | Adds significant complexity; FastAPI reference implementation first | — Pending |

---

*Last updated: 2026-08-21 after initialization*