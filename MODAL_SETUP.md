# PersonaPlex Modal Deployment Guide

Deploy PersonaPlex to Modal's serverless GPUs with WebSocket support.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **HuggingFace Account**: Sign up at [huggingface.co](https://huggingface.co)
3. **HuggingFace Token**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. **Model License**: **REQUIRED** - Accept the gated model license (see below)
5. **DeepSeek API Key** (optional): For smart routing feature - get from [platform.deepseek.com](https://platform.deepseek.com)

## Step 1: Accept HuggingFace Model License (REQUIRED)

PersonaPlex is a **gated model** - you must request and receive access before downloading.

1. Go to: https://huggingface.co/nvidia/personaplex-7b-v1
2. Click **"Agree and access repository"** button
3. Fill in the form (company name, use case, etc.)
4. **Wait for approval** - NVIDIA may take a few hours to approve
5. Once approved, you'll see "You have been granted access to this model"

> ⚠️ **Common Error**: If you see `403 Forbidden` or `GatedRepoError`, you haven't been granted access yet. Check your HuggingFace email or revisit the model page to see your access status.

## Step 2: Create HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Name: `personaplex-modal` (or anything)
4. Type: **Read** (sufficient for downloading models)
5. Click **"Create token"**
6. **Copy the token** (starts with `hf_`)

## Quick Start

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate (opens browser)
modal setup

# 3. Create HuggingFace secret
modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN_HERE

# 4. Create DeepSeek secret (optional - for smart routing)
modal secret create deepseek-secret DEEPSEEK_API_KEY=sk-YOUR_KEY_HERE

# 5. Start development server (from this directory)
cd /Users/mu/src/personaplex
modal serve modal_app.py

# Output:
# ✓ Created web function PersonaPlexServer.serve
# ✓ Serving at https://YOUR_USERNAME--personaplex-server.modal.run
```

## WebSocket URL

After running `modal serve`, your WebSocket URL is:

```
wss://YOUR_USERNAME--personaplex-server.modal.run/api
```

Use this URL in Transgate's `.env.local`:

```bash
PERSONAPLEX_WS_URL=wss://YOUR_USERNAME--personaplex-server.modal.run/api
```

## Commands

| Command | Description |
|---------|-------------|
| `modal serve modal_app.py` | Development mode (hot reload, temporary URL) |
| `modal deploy modal_app.py` | Production deploy (permanent URL) |
| `modal app logs personaplex` | View logs |
| `modal app stop personaplex` | Stop all containers |

## Cost Estimate

| GPU | Cost/Hour | First Request | Warm Request |
|-----|-----------|---------------|--------------|
| A10G (24GB) | ~$0.46 | ~60s (model load) | ~50ms |
| A100 (40GB) | ~$1.10 | ~45s | ~30ms |

**Container Idle Timeout**: 5 minutes (configurable in `modal_app.py`)

After 5 minutes of inactivity, container shuts down → next request has cold start.

## Secret Management

### Create HuggingFace Secret

```bash
# Via CLI
modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxx

# Or via Dashboard
# 1. Go to https://modal.com/secrets
# 2. Click "Create Secret"
# 3. Name: huggingface-secret
# 4. Add key: HF_TOKEN = your_token
```

### Create DeepSeek Secret (Optional - for Smart Routing)

DeepSeek is used for smart routing - when PersonaPlex defers complex product questions with `!!!` marker, DeepSeek provides accurate answers about Transgate.

**Step 1: Get DeepSeek API Key**
1. Go to: https://platform.deepseek.com
2. Sign up / Log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `sk-`)

**Step 2: Create Modal Secret**

```bash
# Via CLI
modal secret create deepseek-secret DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxx

# Or via Dashboard
# 1. Go to https://modal.com/secrets
# 2. Click "Create Secret"
# 3. Name: deepseek-secret
# 4. Add key: DEEPSEEK_API_KEY = your_key
```

**Step 3: Enable in modal_app.py**

After creating the secret, uncomment this line in `modal_app.py`:

```python
secrets=[
    modal.Secret.from_name("huggingface-secret"),
    modal.Secret.from_name("deepseek-secret"),  # <-- Uncomment this line
],
```

> **Note**: Without DeepSeek secret, PersonaPlex will work but defer responses will show a fallback message ("I'm unable to look that up right now").

### Verify Secrets

```bash
modal secret list
# Should show:
#   huggingface-secret
#   deepseek-secret (if created)
```

## Testing the WebSocket

### Using wscat

```bash
# Install wscat
npm install -g wscat

# Connect to Modal
wscat -c "wss://YOUR_USERNAME--personaplex-server.modal.run/api"
```

### Using Browser Console

```javascript
const ws = new WebSocket('wss://YOUR_USERNAME--personaplex-server.modal.run/api');

ws.onopen = () => {
  console.log('Connected!');

  // Send TTS inject (0x07 + UTF-8 text)
  const text = "Hello from Modal!";
  const encoder = new TextEncoder();
  const textBytes = encoder.encode(text);
  const message = new Uint8Array(1 + textBytes.length);
  message[0] = 0x07;
  message.set(textBytes, 1);
  ws.send(message);
};

ws.onmessage = (e) => {
  const data = new Uint8Array(e.data);
  const type = data[0];
  console.log('Received:', type === 0x01 ? 'audio' : type === 0x02 ? 'text' : `type ${type}`);
};
```

## Troubleshooting

### Error: `403 Forbidden` / `GatedRepoError`

```
Cannot access gated repo for url https://huggingface.co/nvidia/personaplex-7b-v1/...
Access to model nvidia/personaplex-7b-v1 is restricted
```

**Cause**: You haven't been granted access to the gated model.

**Fix**:
1. Go to https://huggingface.co/nvidia/personaplex-7b-v1
2. Click "Agree and access repository" button
3. Fill in the required form
4. **Wait for NVIDIA to approve** (can take hours)
5. Check your email or revisit the model page
6. Once you see "You have been granted access", try again

### "Secret not found" Error

```bash
# Create the secret
modal secret create huggingface-secret HF_TOKEN=hf_xxxxx
```

### Model Download Slow/Fails

First run downloads ~15GB model weights. If it fails:

1. Check HF_TOKEN is valid
2. Verify license accepted on HuggingFace
3. Check Modal logs: `modal app logs personaplex`

### WebSocket Connection Refused

1. Ensure `modal serve` is running
2. Check URL matches output from `modal serve`
3. Verify port 8998 in `modal_app.py` matches server

### GPU Out of Memory

A10G has 24GB VRAM. If OOM:

1. Reduce batch size in server.py
2. Use A100 (40GB) instead: change `gpu="A10G"` to `gpu="A100"` in modal_app.py

## Architecture

```
Your Mac                          Modal Cloud
─────────                         ───────────
modal_app.py ──modal serve──→   GPU Container (A10G)
    │                                  │
    │                                  ├─ Loads PersonaPlex model
    │                                  ├─ Runs aiohttp WebSocket server
    │                                  └─ Port 8998
    │                                        │
    │                                        ▼
    │                           Modal Proxy (SSL termination)
    │                                        │
    └──────────────────────────────────────────
                WebSocket URL: wss://...modal.run/api
```

## Files

```
/Users/mu/src/personaplex/
├── modal_app.py           # Modal deployment config (this file)
├── MODAL_SETUP.md         # This guide
├── moshi/
│   └── moshi/
│       ├── server.py      # WebSocket server (modified with TTS inject)
│       └── models/
│           └── lm.py      # LM with step() for TTS injection
└── PERSONAPLEX_SOLUTION.md  # Symlink to Transgate docs
```

## Production Deployment

```bash
# Deploy (creates permanent URL)
modal deploy modal_app.py

# Your production URL:
# wss://YOUR_USERNAME--personaplex-server.modal.run/api

# Update Transgate .env.production
PERSONAPLEX_WS_URL=wss://YOUR_USERNAME--personaplex-server.modal.run/api
```

## Scaling Notes

- **Concurrency**: Each container handles 1 connection (stateful LM)
- **Auto-scaling**: Modal spawns new containers for concurrent users
- **Max containers**: Default 10, increase via Modal dashboard
- **Cold start**: ~60s for first request (model loading)
- **Warm request**: ~50ms when container already running

For high traffic, consider:
1. Increasing `scaledown_window` in modal_app.py (keep containers warm longer)
2. Using `min_containers=1` to always have one warm (costs ~$330/month for A10G)
