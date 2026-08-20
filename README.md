# Post-Audit-Mirage

Identifiability Limits and Occupancy Shift in Lifecycle Auditing

This repository currently implements the first dataset phase through one exactly solvable authorization environment.
It does not yet implement deployment decision methods or support paper claims.

## What AuthGate-v0 contains

AuthGate-v0 is a 14-state environment with a maximum horizon of ten steps.
The safe and harmful worlds produce the same offline audit observations but differ after temporary access has been granted.
The candidate has 2 percent audit harm in both worlds, 2 percent live harm in the safe world, and 20 percent live harm in the harmful world for approved calibration requests.

The authoritative states, transitions, policies, splits, and labels are ordinary deterministic Python code.
GPT-5.6 Luna supplies only organization names, resource names, and request wording.

## Environment

The project uses only the Python 3.13 standard library.

```bash
python3 -m venv .venv
.venv/bin/python -m unittest discover -s tests -v
```

## Generate surface variants

Set `OPENROUTER_API_KEY` in the shell without saving it in this repository.
The command makes twelve structured requests to `openai/gpt-5.6-luna` and writes three variants per semantic template.
Provider fallback is disabled.
Each row records whether OpenRouter returned the requested alias or its dated canonical identity.

```bash
.venv/bin/python -m post_audit_mirage generate-surfaces
```

## Build the dataset

```bash
.venv/bin/python -m post_audit_mirage build-dataset
```

The generated files are:

- `data/authgate_v0/surface_variants.jsonl` with 36 model-written surface cases.
- `data/authgate_v0/audit.jsonl` with 72 policy-specific audit records.
- `data/authgate_v0/truth.jsonl` with 144 policy and world truth records.

Audit rows contain no world label, post-audit state, occupancy, or live-harm value.
Truth rows are kept separate and are calculated with exact rational arithmetic.

## End-to-end tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests invoke the real command-line workflow against a local HTTP server shaped like OpenRouter.
They verify the complete output, exact 2 percent and 20 percent calibration, policy actions, split separation, paired audit equality, deterministic rebuilding, and invalid-response cleanup without spending API credit.
