"""Serve gpt-oss-20b on Modal with vLLM's OpenAI-compatible API (run on your laptop, not in this workspace).

    modal secret create vllm-key VLLM_API_KEY=<long random string>
    modal deploy modal_jobs/serve_gptoss.py

Modal prints a URL like https://<workspace>--swarm-inv-gptoss-serve.modal.run. Add /v1 to it for
VLLM_BASE_URL_GPTOSS in compute.env, and requests for "vllm/gptoss" go there. The first request downloads the
weights (about 14 GB) and can take 5 to 10 minutes; later starts reuse the cached volume. The container scales to
zero after 10 idle minutes, and `modal app stop swarm-inv-gptoss` stops it at once. Billing is per GPU-second (one
H100).

Written for Modal 1.x (`@modal.concurrent`, `scaledown_window`). If `vllm==0.10.2` fails to build or does not know
gpt-oss (it needs vLLM 0.10.1 or newer), bump it to a newer release. Nothing else in this repo depends on the
version.

Adapted from collusion-sting modal_jobs/serve_gptoss.py at 4be9164. The app name, this docstring (whose deploy
command named serve_qwen.py there) and the MAX_MODEL_LEN constant are new. The image, volumes, secret and server
flags are unchanged.
"""

import subprocess

import modal

MODEL = "openai/gpt-oss-20b"  # Apache-2.0; MoE with about 3.6B active parameters, fits one H100 easily
MINUTES = 60
# Prompt plus output, in tokens, and gpt-oss's reasoning counts as output. Longer requests get HTTP 400. The model
# handles up to 131,072, so raise this and redeploy if investigator prompts outgrow it.
MAX_MODEL_LEN = 16384

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.10.2",
    # vLLM 0.10.2 needs transformers>=4.55.2 (PyPI metadata) and predates the 5.x line, so cap below 4.57.
    # 4.56.x registers gpt_oss.
    "transformers>=4.55.2,<4.57",
    "hf-transfer==0.1.9",
)
image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
app = modal.App("swarm-inv-gptoss")


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
