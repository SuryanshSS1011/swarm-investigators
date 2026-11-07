"""Serve Qwen3-32B-FP8 on Modal with vLLM's OpenAI-compatible API (run on your laptop, not in this workspace).

    modal secret create vllm-key VLLM_API_KEY=<long random string>
    modal deploy modal_jobs/serve_qwen.py

Modal prints a URL like https://<workspace>--swarm-inv-qwen-serve.modal.run. Add /v1 to it for VLLM_BASE_URL_QWEN in
compute.env, and requests for "vllm/qwen" go there. The first request downloads the weights (about 33 GB) and can
take 5 to 10 minutes; later starts reuse the cached volume. The container scales to zero after 10 idle minutes, and
`modal app stop swarm-inv-qwen` stops it at once. Billing is per GPU-second (one H100).

Written for Modal 1.x (`@modal.concurrent`, `scaledown_window`). If `vllm==0.9.2` fails to build or does not know
Qwen3, bump it to a newer release; nothing else in this repo depends on the version.

Adapted from collusion-sting modal_jobs/serve_qwen.py at 4be9164. The app name, this docstring and the
MAX_MODEL_LEN constant are new. The image, volumes, secret and server flags are unchanged.
"""

import subprocess

import modal

MODEL = "Qwen/Qwen3-32B-FP8"  # Apache-2.0, ungated; FP8 weights fit one H100 with room for the KV cache
MINUTES = 60
# Prompt plus output, in tokens. Longer requests get HTTP 400. Qwen3 handles up to 32,768 without YaRN, so raise
# this and redeploy if investigator prompts outgrow it.
MAX_MODEL_LEN = 16384

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.9.2",
    # vLLM 0.9.2 registers its own "aimv2" config, which transformers added in 4.54 (checked against
    # 4.53.3, 4.54.1 and later): an unpinned transformers crashes vLLM on import. 4.53.x has Qwen3.
    "transformers>=4.53.2,<4.54",
    "hf-transfer==0.1.9",
)
image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("swarm-inv-qwen")


@app.function(
    image=image,
    gpu="H100",
    timeout=12 * 60 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("vllm-key")],
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * MINUTES)
def serve():
    import os

    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        "0.92",
        "--api-key",
        os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
