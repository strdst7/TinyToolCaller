# Roadmap: TinyToolCaller — Production Deployment Infrastructure

## Overview

Transform the TinyToolCaller research codebase (training pipeline, evaluation suite, publication artifacts) into a production-grade inference service. Start with a FastAPI server and the §22.1 validation safety stack, harden with auth hooks and audit logging, package with Docker and CI/CD, then document the whole system so any application can deploy TinyToolCaller as a structured tool-calling component.

## Phases

- [ ] **Phase 1: Core Server + Validation Stack** — FastAPI inference server with the 6-layer validation stack, CLI mode, codebase cleanup
- [ ] **Phase 2: Production Hardening** — Authorization hook, audit logging, health/readiness endpoints, environment-variable configuration
- [ ] **Phase 3: CI/CD + Docker** — Multi-stage Dockerfile (GPU + CPU), docker-compose, GitHub Actions CI/CD to GHCR
- [ ] **Phase 4: Documentation + Diagrams** — Deployment guide, configuration reference, production docs, architecture diagram

## Phase Details

### Phase 1: Core Server + Validation Stack
**Goal**: A working FastAPI inference server that accepts tool-calling requests, runs them through the §22.1 validation stack (extract → allowlist → schema → retry-with-repair), and returns structured responses. CLI mode for local/script usage. Codebase cleaned of orphaned files and dependency versions locked.

**Depends on**: Nothing (first phase)

**Requirements**: API-01, API-02, API-03, API-04, API-06, API-07, VAL-01, VAL-02, VAL-03, VAL-05, DOC-04, CLN-01, CLN-02

**Success Criteria** (what must be TRUE):
  1. User can start the server with `python -m tinytoolcaller.serve` and it loads the model at startup (no per-request loading delay)
  2. User can `POST /tool-call` with `{"request": "...", "tools": [...]}` and receive `{"name": "...", "arguments": {...}}` on success
  3. User receives `{"error": "unparseable"}` when the model output cannot be parsed as JSON (6-layer validation: extract → allowlist → schema checks applied)
  4. User can run one-shot inference via CLI: `python -m tinytoolcaller.serve --model <path> --prompt "..." --tools [...]`
  5. User sees auto-generated OpenAPI docs at `/docs` with request/response schemas documented
  6. The orphaned `TinyToolCaller-files/` directory is removed from the repo and `requirements.txt` pins all dependency versions

**Plans**: 5 plans

Plans:
- [ ] 01-01: FastAPI application scaffold — server entry point, `/tool-call` route, Pydantic request/response models, model loading at startup, configurable port via CLI arg
- [ ] 01-02: Validation stack — `extract_json` integration, tool allowlist check, JSON Schema validation, retry-with-repair loop (layers 1-5 of §22.1)
- [ ] 01-03: CLI inference mode — `python -m tinytoolcaller.serve` with `--model`, `--prompt`, `--tools` flags for one-shot inference without server
- [ ] 01-04: OpenAPI docs — ensure FastAPI auto-generates `/docs` with proper request/response schemas, validation error examples, and status codes
- [ ] 01-05: Codebase cleanup — remove orphaned `TinyToolCaller-files/` directory, pin dependency versions in `requirements.txt`, verify existing 41-test suite still passes

**UI hint**: yes

### Phase 2: Production Hardening
**Goal**: The server becomes deployment-ready with an authorization hook for per-request scoping, structured audit logging following the §23 log schema, health/readiness endpoints for Kubernetes probes, and all configuration driven by environment variables.

**Depends on**: Phase 1

**Requirements**: API-05, VAL-04, VAL-06

**Success Criteria** (what must be TRUE):
  1. User can configure model path, port, allowlist, and log level via environment variables (no code changes needed)
  2. User can inject an authorization callable that scopes each request by user, tool, and argument before execution
  3. Every request is logged with the §23 log schema (timestamp, request_id, tool_name, validation_result, latency_ms, auth_context)
  4. Kubernetes probes can hit `/health` (server alive) and `/ready` (model loaded, accepting traffic) and get appropriate HTTP 200/503 responses

**Plans**: 3 plans

Plans:
- [ ] 02-01: Environment-variable configuration — wire env vars for model path, port, allowlist path, log level; add `.env.example`; validate at startup
- [ ] 02-02: Authorization hook — injectable callable interface (per-user, per-tool, per-argument scoping); integrate into validation stack as layer 4; no-op default for backward compatibility
- [ ] 02-03: Health/readiness endpoints + audit logging — `/health` and `/ready` routes; structured JSON logging with §23 schema; request_id propagation through each request lifecycle

### Phase 3: CI/CD + Docker
**Goal**: The server is packaged as a Docker image (GPU + CPU variants), deployable locally with `docker compose up`, and automatically built and published to GitHub Container Registry via GitHub Actions on every merge to main.

**Depends on**: Phase 2

**Requirements**: DOCK-01, DOCK-02, DOCK-03, DOCK-04, DOCK-05, CICD-01, CICD-02, CICD-03

**Success Criteria** (what must be TRUE):
  1. User can run `docker compose up` from the repo root and have the server listening on the configured port (CPU variant for local dev)
  2. User can build the GPU variant with `docker build --target gpu -t tinytoolcaller:gpu .` — CUDA + bitsandbytes included
  3. `.dockerignore` excludes tests/, scripts/, .git/, .planning/ — the image contains only what the server needs
  4. On every PR to `main`, GitHub Actions runs lint + test + Docker build (CI succeeds before merge)
  5. On every merge to `main`, GitHub Actions builds both variants and pushes to `ghcr.io/<owner>/tinytoolcaller:*` tagged with git SHA and semantic version

**Plans**: 3 plans

Plans:
- [ ] 03-01: Multi-stage Dockerfile — base Python stage → deps stage → CPU target → GPU target (CUDA + bitsandbytes); `.dockerignore`; `docker compose.yml` for one-command local dev
- [ ] 03-02: GitHub Actions CI workflow — trigger on PR to main: lint (flake8) → test (pytest) → build Docker (CPU variant for speed); fail fast on any step
- [ ] 03-03: GitHub Actions CD workflow — trigger on merge to main: build CPU + GPU variants; tag with `git-sha`, `vX.Y.Z` (semantic); push to GHCR

### Phase 4: Documentation + Diagrams
**Goal**: The repo is fully documented for new users and production operators. README has a quick-start with `docker compose up`, complete configuration reference, and production deployment guide covering Kubernetes, monitoring, and alerting. An architecture diagram documents the deployment stack.

**Depends on**: Phase 3

**Requirements**: DOC-01, DOC-02, DOC-03, CLN-03

**Success Criteria** (what must be TRUE):
  1. User can follow "Quick Start — Run the Server" in README and have the server running in under 2 minutes with `docker compose up`
  2. User can find every environment variable documented with its purpose, default value, and example in the "Configuration" section
  3. User can follow "Production Deployment" guidance to deploy on Kubernetes with health probes, resource limits, and monitoring hooks
  4. User can view the architecture diagram (at `docs/architecture.png` or similar) showing the request flow: client → FastAPI → validation stack → model → response, plus the Docker/CI/CD layers

**Plans**: 2 plans

Plans:
- [ ] 04-01: README deployment documentation — Quick Start, Configuration reference (all env vars), Production Deployment guide (Kubernetes, monitoring, alerting), API reference pointing to `/docs`
- [ ] 04-02: Architecture diagram — update `docs/architecture.png` (or equivalent diagram-as-code in `scripts/`) showing deployment architecture: request flow, validation layers, Docker build stages, CI/CD pipeline

## Progress

**Execution Order:** Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Server + Validation Stack | 0/5 | Not started | - |
| 2. Production Hardening | 0/3 | Not started | - |
| 3. CI/CD + Docker | 0/3 | Not started | - |
| 4. Documentation + Diagrams | 0/2 | Not started | - |