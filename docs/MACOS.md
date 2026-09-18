# Apple Silicon direct scoring

OpenJev's experimental macOS path runs direct typed-option scoring through PyTorch MPS. It does not reproduce the published Qwen3.5-4B RTX 3090 measurements.

## Requirements

- Apple Silicon with MPS available
- Python 3.10 through 3.13; the pinned NumPy 2.2.6 release has no Python 3.14 wheel
- Enough free storage for the Python environment and the approximately 1.5 GB Qwen3-0.6B BF16 checkpoint

## Install

Use the repository's exact dependency versions and refuse source distributions:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --only-binary=:all: -c constraints-macos.txt -r requirements.txt
.venv/bin/pip install --no-deps --no-build-isolation -e .
```

## Run

The output path must not already exist:

```bash
.venv/bin/openjev-score --mode direct --device mps \
  --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --input examples/decisions.jsonl \
  --output results-qwen3-06b-mps.jsonl
```

The first run downloads the pinned model revision into the Hugging Face cache. Model execution uses FP16 on MPS. Unsupported MPS operations fail rather than falling back silently to CPU.

## Boundaries

- MPS and CPU support only `direct` mode.
- `serial`, `shared`, and `reranker` remain CUDA-only.
- Returned probabilities are conditional on the displayed options and are not calibrated confidence.
- Qwen3-0.6B quality and timing are not comparable to the published Qwen3.5-4B CUDA evidence.
- The published results and checksums remain unchanged.
