# Qwen3-0.6B Apple Silicon Direct Scoring Design

## Objective

Add a Mac-native direct-scoring path to the OpenJev command-line tool and prove it by running the owned example workload with a pinned Qwen3-0.6B model on Apple MPS. Preserve the existing typed option-logit contract and leave all published Qwen3.5-4B CUDA evidence unchanged.

## Scope

The supported Apple Silicon path is:

```text
openjev-score --mode direct --device mps
```

The model used for the acceptance run is `Qwen/Qwen3-0.6B` at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`.

The change also adds `--device auto|cuda|mps|cpu`. `auto` selects CUDA when exactly one CUDA device is visible. If CUDA is available with more than one visible device, selection fails and preserves the existing single-GPU contract; if CUDA is unavailable, `auto` selects MPS when available and otherwise CPU. Explicit device requests fail if unavailable. There is no silent runtime fallback after selection.

MPS support is limited to direct scoring in this change. Serial prefix reuse, parallel shared-prefix scoring, and reranker scoring remain CUDA-only until their cache and numerical behavior are independently validated on MPS.

## Non-goals

- Do not reproduce or modify Jev's undisclosed architecture, sampler, or training.
- Do not add MLX, change dependency versions, or introduce another inference backend.
- Do not modify `results/phase1-summary.json`, committed raw reports, checksums, or headline claims.
- Do not claim that Qwen3-0.6B quality or timing is comparable to the published Qwen3.5-4B RTX 3090 measurements.
- Do not open an upstream pull request.
- Do not add MPS support for `serial`, `shared`, or `reranker` modes.

## Runtime architecture

Create `src/openjev_phase1/runtime.py` as the single hardware-policy boundary. It will:

- resolve `auto`, `cuda`, `mps`, and `cpu` requests;
- enforce the existing exactly-one-visible-CUDA-device rule;
- select BF16 for CUDA, FP16 for MPS, and FP32 for CPU;
- synchronize CUDA with `torch.cuda.synchronize` and MPS with `torch.mps.synchronize` for accurate timing;
- expose a validation function that rejects unsupported mode/device combinations.

`src/openjev_phase1/core.py` will use this policy when loading a causal model. Remote models will continue to require a pinned 40-character commit revision, `trust_remote_code` will remain false, and incomplete checkpoints will still be rejected. CUDA will retain the existing device-map loading path. MPS and CPU will load on the host and then move the model to the resolved device using the selected dtype.

Model metadata will include the requested device, resolved device, dtype, Torch version, Transformers version, model source, and immutable model revision. This makes an MPS output distinguishable from the published CUDA evidence.

`src/openjev_phase1/direct.py` will replace CUDA-specific synchronization with the runtime helper. Prompt construction, single-token answer-slot validation, no-truncation behavior, last-position vocabulary readout, option-logit selection, and conditional softmax remain unchanged.

`src/openjev_phase1/cli.py` will add the `--device` argument, validate the mode/device combination before model loading, and pass the resolved request into the loader. Existing commands without `--device` use `auto`.

## Data flow

```text
JSONL input
  -> validate state, question, and typed options
  -> resolve device and dtype
  -> load the pinned Qwen3-0.6B checkpoint
  -> render the existing direct-decision prompt
  -> verify A-P answer labels are exact single tokens at the prompt boundary
  -> execute one model forward pass per decision
  -> select only the declared option-token logits
  -> normalize those logits across the supplied options
  -> write create-only JSONL results with runtime metadata and timing
```

The returned values remain conditional option scores. Documentation must continue to state that they are not calibrated confidence.

## Error handling

- An unavailable explicit device produces an actionable error naming the requested backend.
- More than one visible CUDA device remains an error.
- MPS with `serial`, `shared`, or `reranker` mode is rejected before downloading or loading a model.
- CPU with those modes is rejected for the same reason.
- Model revisions that are not 40-character hexadecimal commits remain invalid for remote models.
- A tokenizer whose option labels are not exact single tokens remains invalid.
- Inputs exceeding `--max-tokens` remain errors; no truncation is introduced.
- Unsupported MPS model operations surface as errors. The tool must not silently retry on CPU.
- Existing output paths remain errors so experimental evidence is never overwritten.

## Files

Planned source and test changes:

- Create `src/openjev_phase1/runtime.py` for device, dtype, synchronization, and compatibility policy.
- Modify `src/openjev_phase1/core.py` to load models through that policy and record runtime metadata.
- Modify `src/openjev_phase1/direct.py` to use backend-neutral synchronization.
- Modify `src/openjev_phase1/cli.py` to expose and validate `--device`.
- Create `tests/test_runtime.py` for pure device-policy and synchronization tests using fakes.
- Create `tests/test_cli.py` for parser coverage and pre-load rejection of unsupported mode/device combinations.
- Modify `tests/test_core.py` only where loader metadata or signatures require coverage.
- Modify `README.md` with a concise Apple Silicon quick-start link.
- Create `docs/MACOS.md` with the pinned Qwen3-0.6B command, limitations, expected download size, and evidence boundary.

No manifest or published result file will change. If implementation reveals that another file is necessary, stop and revise the plan before editing it.

## Dependency policy

No dependencies will be added or updated. The existing exact versions are retained:

- `torch==2.10.0`
- `transformers==5.17.0`
- `accelerate==1.12.0`
- `safetensors==0.8.0`
- `huggingface-hub==1.31.0`
- `tokenizers==0.23.2`
- `numpy==2.2.6`
- `sentencepiece==0.2.1`
- `protobuf==7.36.1`
- `pytest==8.4.2`

All versions were more than seven days old at the 2026-09-17 dependency audit and publish compatible binary wheels. Installation must refuse third-party source distributions:

```bash
pip install --only-binary=:all: -r requirements.txt
pip install --no-deps -e .
```

The second command builds only the reviewed local package. No lockfile will be regenerated or materially rewritten.

## Test strategy

Before implementation, run the complete upstream suite in the isolated environment to establish the baseline.

Use test-driven development for:

- `auto` device precedence;
- explicit unavailable-device failures;
- CUDA/BF16, MPS/FP16, and CPU/FP32 dtype selection;
- CUDA and MPS synchronization dispatch;
- direct mode acceptance on MPS and CPU;
- pre-load rejection of unsupported mode/device pairs;
- runtime metadata fields;
- unchanged direct-score probability normalization.

After implementation, run:

```bash
pytest -q
(cd results/raw && sha256sum -c SHA256SUMS)
python benchmarks/verify_published.py
```

Then run the live MPS acceptance command with a new output path:

```bash
openjev-score --mode direct --device mps \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --input examples/decisions.jsonl \
  --output results-qwen3-06b-mps.jsonl
```

Validate that all three owned example rows:

- complete without fallback;
- contain finite option logits and probabilities;
- have probabilities summing to one within floating-point tolerance;
- report `mps`, `float16`, and the pinned revision in model metadata;
- preserve input option IDs and prompt hashes;
- contain per-row forward and total timings.

The local result file is not committed as published evidence. The completion report will provide model-load time, per-row timings, selected options, and any numerical or runtime caveats.

## Git workflow

Development occurs in `crocodiledundalk/openjev` on `feat/qwen3-06b-mps`, with `TheoLeeCJ/openjev` retained as the `upstream` remote. Commits use conventional prefixes and remain narrowly scoped. The finished branch is pushed to the fork, but no upstream pull request is created.
