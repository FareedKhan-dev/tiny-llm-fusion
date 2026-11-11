# tiny-llm-fusion

Four small open models, fused into one self-improving engine and measured against Claude Opus 5.5.

The team is Qwen3.5-4B, Llama-3.2-3B-Instruct, Gemma-4-E2B-it and Phi-4-mini-instruct, one model per NVIDIA L40S 48GB VM. The engine adds a GPU control plane, vLLM serving behind one gateway, a difficulty-aware router, tools (a sandbox, program-aided reasoning and BM25 retrieval), parallel sampling, Mixture-of-Agents, a weighted vote, a verifier, calibration and a DPO loop. This is the code of the post "Building a Production-Grade Fusion of Tiny LLMs That Rivals Claude Opus 5.5".

## Layout

```
tiny-llm-fusion/
├── fusion/
│   ├── config.py      # models, prices, budgets
│   ├── data.py        # benchmarks, the seal, the pools
│   ├── platform/      # fleet, serve, bootstrap.sh, gateway, telemetry, governor, episodes
│   ├── tools/         # sandbox, graders, pal, retrieval, frontier
│   ├── engine/        # pool, sampler, cache, vote, moa, verifier, calibrate, router
│   ├── learn/         # improve
│   ├── serve_app/     # orchestrator
│   └── eval/          # harness
├── data/              # benchmark files and the sealed manifest
└── results/           # every number the post prints
```

## Setup

The code runs on a small CPU box, and the control plane rents one GPU VM per model.

```bash
uv sync                                            # add --extra train on the box that runs DPO
export GPU_CLOUD_URL="https://your-gpu-cloud/v1"   # any VM API that can create, read and delete
export GPU_CLOUD_KEY="your key"
export GATEWAY_KEY="a long random string"          # every model call carries it
export HF_TOKEN="your token"                       # Llama 3.2 is a gated download
export ANTHROPIC_API_KEY="your key"                # only the frontier yardstick calls Claude
uv run jupyter lab
```

The drivers are cells of one notebook, run in the order of the post: the data and the seal, the fleet, the pool, the Opus pilot, the tools, sampling and the vote, the aggregator, the verifier and calibration, the router, the orchestrator, DPO, and last the sealed evaluation.

## Results

`results/` holds the numbers the post prints. Each value is a `{"v": ..., "provisional": ...}` leaf, and a value stays marked provisional until a full run replaces it.
