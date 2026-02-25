"""
PersonaPlex Modal Deployment

Run locally for development:
    modal serve modal_app.py

Deploy to production:
    modal deploy modal_app.py

Your WebSocket URL will be:
    wss://<your-username>--personaplex-server.modal.run/api

Requirements:
    pip install modal
    modal setup  # One-time authentication
"""

import modal
import os

# =============================================================================
# Modal App Configuration
# =============================================================================

app = modal.App("personaplex")

# Persistent volume for HuggingFace model cache (survives container restarts)
model_cache = modal.Volume.from_name("personaplex-hf-cache", create_if_missing=True)

# GPU image with all dependencies + local moshi package
# NOTE: Modal 1.0+ uses add_local_dir() instead of Mount
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libopus-dev",
        "libopusfile-dev",
        "git",
        "ffmpeg",
    )
    .pip_install(
        "torch>=2.0.0",
        "torchaudio",
        "numpy",
        "aiohttp",
        "sentencepiece",
        "huggingface_hub",
        "sphn",  # Opus streaming
    )
    # Copy local moshi directory into container (copy=True required for pip install)
    .add_local_dir("./moshi", "/root/moshi", copy=True)
    # Install moshi package from copied directory
    .run_commands("pip install /root/moshi")
)

# =============================================================================
# Secrets Setup
# =============================================================================
#
# Create these secrets in Modal Dashboard (https://modal.com/secrets):
#
# 1. huggingface-secret (REQUIRED):
#    - HF_TOKEN: Your HuggingFace token
#    - Get token: https://huggingface.co/settings/tokens
#    - Accept license: https://huggingface.co/nvidia/personaplex-7b-v1
#    CLI: modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxx
#
# 2. deepseek-secret (OPTIONAL - for smart routing):
#    - DEEPSEEK_API_KEY: Your DeepSeek API key
#    - Get key: https://platform.deepseek.com
#    CLI: modal secret create deepseek-secret DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx
#
# =============================================================================


# =============================================================================
# PersonaPlex Server Class
# =============================================================================

@app.cls(
    image=image,
    gpu="A10G",  # 24GB VRAM, ~$0.46/hr (or use "A100" for faster, ~$1.10/hr)
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        # Uncomment below line after creating deepseek-secret for smart routing:
        # modal.Secret.from_name("deepseek-secret"),
    ],
    timeout=900,  # 15 minutes max per connection
    scaledown_window=300,  # Keep warm for 5 min after last request
    volumes={"/root/.cache/huggingface": model_cache},  # Persistent model cache
)
@modal.concurrent(max_inputs=1)  # One conversation per container (stateful)
class PersonaPlexServer:
    """
    PersonaPlex WebSocket server running on Modal GPU.

    Each container handles ONE WebSocket connection at a time because
    PersonaPlex maintains conversation state (LM context, audio buffers).
    """

    @modal.enter()
    def load_model(self):
        """Called once when container starts - load model into GPU memory."""
        import torch

        print("Loading PersonaPlex model...")

        # Set HF token from secret
        os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

        # Load model (downloads ~15GB on first run, cached after)
        self.device = torch.device("cuda")

        # These will be initialized per-connection in the WebSocket handler
        # because they maintain streaming state
        self.model_loaded = True
        print("Model loading deferred to connection time for state isolation")

    @modal.web_server(port=8998, startup_timeout=600)  # 10 min for first model download
    def serve(self):
        """
        Starts the aiohttp WebSocket server.

        Modal routes requests to wss://<app>--personaplex-server.modal.run
        to this server's port 8998.

        NOTE: No SSL here - Modal handles SSL termination at the edge.
        """
        import subprocess
        import sys

        # Run the PersonaPlex server WITHOUT SSL
        # Modal's proxy handles SSL termination externally
        cmd = [
            sys.executable, "-m", "moshi.server",
            # No --ssl flag = HTTP/WS mode (Modal handles SSL at edge)
            "--host", "0.0.0.0",
            "--port", "8998",
        ]

        print(f"Starting PersonaPlex server: {' '.join(cmd)}")
        subprocess.Popen(cmd, env={**os.environ})


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "service": "personaplex"}


# =============================================================================
# Local Development Entry Point
# =============================================================================

@app.local_entrypoint()
def main():
    """
    Local development entry point.

    Run with: modal run modal_app.py

    This just prints the deployment URL - actual server runs in cloud.
    """
    print("\n" + "=" * 60)
    print("PersonaPlex Modal Deployment")
    print("=" * 60)
    print("\nTo start the development server:")
    print("  modal serve modal_app.py")
    print("\nTo deploy to production:")
    print("  modal deploy modal_app.py")
    print("\nYour WebSocket URL will be:")
    print("  wss://<your-username>--personaplex-server.modal.run/api")
    print("\n" + "=" * 60 + "\n")
