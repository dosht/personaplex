# Smart Routing for Full-Duplex Speech Models: Augmenting PersonaPlex with External LLM Knowledge

## Abstract

Full-duplex speech-to-speech models like Moshi and PersonaPlex achieve remarkably low latency (~200ms) conversational interaction by jointly modeling user and agent audio streams. However, these models use relatively small language model backbones (7B parameters) that are optimized for conversational dynamics -- turn-taking, interruptions, backchannels -- rather than factual knowledge or instruction following. This creates a fundamental tension: the model excels at *sounding* like a helpful agent but struggles to *be* one when domain-specific knowledge is required.

This document describes an experimental architecture we call **Smart Routing**, in which PersonaPlex handles the real-time voice layer while an external LLM (DeepSeek) provides intelligent responses to domain-specific questions. We implemented a complete prototype including a new WebSocket protocol extension (TTS injection via message type `0x07`), server-side defer detection, asynchronous LLM consultation, and forced text-token generation through the existing model. We report on what worked, what did not, and what directions remain open.

## 1. Problem Statement

We wanted to build a voice AI assistant for a product landing page -- a widget where visitors could have a real-time spoken conversation about the product's features, pricing, and capabilities. The requirements were:

1. **Low latency**: responses under 500ms for natural conversation flow
2. **Full-duplex**: the user should be able to interrupt, and the agent should produce backchannels ("uh-huh", "right") naturally
3. **Domain knowledge**: accurate answers about specific product details (pricing tiers, feature lists, supported languages, compliance certifications)
4. **Natural voice**: human-like prosody, not robotic TTS

PersonaPlex satisfies requirements 1, 2, and 4 out of the box. It fails on requirement 3. When asked "What are your pricing plans?", it generates fluent but fabricated answers -- or worse, confidently wrong ones. The Helium 7B backbone was trained for conversational realism, not factual grounding.

The core research question: **Can we route specific queries from a full-duplex speech model to an external LLM and inject the response back into the speech stream, without breaking the real-time conversational experience?**

## 2. Background and Related Work

### 2.1 The Moshi Architecture

PersonaPlex is built on the Moshi architecture (Defossez et al., 2024). Moshi models spoken dialogue as parallel audio token streams using a neural audio codec (Mimi). The key architectural features relevant to this work:

- **Dual-stream modeling**: The 7B transformer processes two simultaneous audio streams (agent output and user input) as interleaved codebook tokens, allowing full-duplex interaction without explicit turn-taking.
- **Inner Monologue**: Text tokens are generated as a prefix to audio tokens at each step, improving linguistic quality. These text tokens form a time-aligned transcript that we can inspect for routing signals.
- **Depformer**: A secondary 6-layer transformer generates dependent audio codebook tokens autoregressively, conditioned on the main transformer's output.
- **Frame rate**: 12.5 Hz (80ms per frame), with 8 audio codebooks per stream using a 2048-bin vocabulary.

The `LMGen.step()` method accepts an optional `text_token` parameter that, when provided, forces the text output while still generating corresponding audio tokens through the depformer. This is the mechanism we exploit for TTS injection.

### 2.2 The Instruction-Following Gap in Speech Models

A central challenge we encountered is well-documented in the broader LLM literature: small models struggle with reliable instruction following, and speech models compound this problem.

Geng et al. (2025) demonstrate in "Control Illusion" that even state-of-the-art text LLMs (70B+ parameters) struggle with consistent instruction prioritization. System/user prompt separation "fails to establish a reliable instruction hierarchy," and models exhibit strong inherent biases toward certain behaviors regardless of prompt instructions. Our experience with a 7B speech model that was never explicitly trained for instruction following confirms this finding in a more extreme setting.

The DiVA approach (Held et al., 2024) addresses a related problem -- Speech LLMs "forgetting" text-only capabilities during supervised fine-tuning -- by using text LLM responses to transcripts as self-supervision. This is conceptually similar to our routing approach: we preserve the speech model's conversational strengths while delegating knowledge-intensive responses to a text LLM. The difference is that DiVA does this at training time through distillation, while we attempt it at inference time through runtime routing.

FunAudioLLM (An et al., 2024) takes a modular approach with separate SenseVoice (understanding) and CosyVoice (generation) models orchestrated by an LLM. This pipeline architecture achieves high quality but at the cost of latency -- exactly the tradeoff full-duplex models like Moshi were designed to avoid.

### 2.3 Hybrid Speech Architectures

The general pattern of combining a fast but limited model with a slow but capable one appears in several forms in recent work:

- **Speculative decoding** uses a small model to draft tokens that a large model verifies, optimizing throughput. Our approach inverts this: the small model handles the real-time stream, and the large model is consulted asynchronously for specific queries.
- **Mixture-of-experts** routes different inputs to different specialized sub-networks. Our routing is coarser-grained: entire conversational turns are routed between two fundamentally different systems.
- **Tool use in LLMs** (e.g., function calling) is the closest analogy. We are effectively trying to give a speech model the ability to "call a tool" (the external LLM) mid-conversation. The difference is that tool-use protocols were designed into text LLMs during training, while we are attempting to bolt this capability onto a model that was not designed for it.

## 3. Architecture

### 3.1 System Overview

The Smart Routing architecture has three components:

```
Browser ← WebSocket (binary) → PersonaPlex Server (GPU)
                                      │
                                      ├── Moshi LM (7B) generates text + audio
                                      ├── Text stream monitored for defer signals
                                      ├── On defer: async call to DeepSeek API
                                      └── DeepSeek response injected via TTS
```

The entire routing pipeline runs server-side. The browser client is unaware of the routing; it receives a continuous stream of audio and text regardless of whether the response originated from PersonaPlex's LM or from DeepSeek via TTS injection. See [WEB_CLIENT.md](./WEB_CLIENT.md) for the full client-side protocol and audio pipeline documentation.

### 3.2 Protocol Extension

We extended the Moshi WebSocket protocol with a new message type:

| Byte | Type | Direction | Payload |
|------|------|-----------|---------|
| `0x00` | Handshake | S→C | Connection ready |
| `0x01` | Audio | Bidirectional | Opus-encoded frames |
| `0x02` | Text | S→C | UTF-8 transcript tokens |
| `0x03` | Control | C→S | Action byte |
| `0x04` | Metadata | Bidirectional | JSON |
| `0x05` | Error | S→C | UTF-8 message |
| `0x06` | Ping | Bidirectional | Echo |
| **`0x07`** | **TTS Inject** | **C→S** | **UTF-8 text to synthesize** |

All messages share the same framing: `[1 byte type][N bytes payload]`, with WebSocket framing providing message boundaries.

### 3.3 TTS Injection Mechanism

The TTS injection exploits the existing `LMGen.step()` API:

```python
async def process_tts_inject(self, text, opus_writer, ws):
    tokens = self.text_tokenizer.encode(text)

    for text_token in tokens:
        silence_frame = torch.zeros(1, 1, frame_size)
        silence_codes = self.mimi.encode(silence_frame)

        out = self.lm_gen.step(
            input_tokens=silence_codes,    # silence on user channel
            text_token=text_token           # forced text token
        )
        # out contains audio tokens generated by the depformer
        # conditioned on the forced text token

        pcm = self.mimi.decode(out[:, 1:9])
        opus_writer.append_pcm(pcm[0, 0].numpy())

        transcript = self.text_tokenizer.id_to_piece(text_token)
        await ws.send_bytes(b"\x02" + transcript.encode())

        await asyncio.sleep(0.001)  # yield to event loop
```

This works because the depformer generates audio tokens that are consistent with the forced text token, using the model's current voice state (pitch, pace, style). The resulting speech sounds natural and matches the voice established during the voice prompt phase.

The injection queue uses `asyncio.Queue(maxsize=10)` for backpressure, checked non-blockingly in the main `opus_loop()` with `get_nowait()`.

### 3.4 Defer Detection

We explored two approaches to signaling when PersonaPlex should defer to DeepSeek.

**Approach 1: Explicit Marker (`!!!`)**

The system prompt instructs PersonaPlex to end responses with `!!!` when it encounters a product question:

```
<system>
You are Transgate's voice assistant.

ROUTING RULES:
1. HANDLE DIRECTLY: greetings, acknowledgments, backchannels
2. DEFER by ending with !!! for: product questions, how-to, comparisons

EXAMPLES:
User: "What's your pricing?"
You: "Great question! Let me tell you about our plans...!!!"
<system>
```

The server accumulates text tokens and checks for the marker:

```python
self.text_buffer += text
if "!!!" in self.text_buffer:
    user_query = extract_last_user_query(conversation_history)
    response = await call_deepseek(user_query, context)
    await process_tts_inject(response)
    self.text_buffer = ""
```

**Approach 2: Natural Phrase Detection**

After observing that the `!!!` marker was unreliable (see Section 5), we attempted detection based on filler phrases that PersonaPlex naturally produces:

```python
DEFER_PHRASES = [
    "let me check",
    "let me find",
    "let me look",
    "let me see",
]
```

The system prompt was simplified to encourage these phrases:

```
<system>
You are Transgate's friendly voice assistant.
For product questions about pricing, features, accuracy, or languages:
Say "Let me check that for you" then stop talking.
Keep responses brief.
<system>
```

### 3.5 Conversation Context Tracking

To provide DeepSeek with conversational context, the server tracks recent utterances:

```python
conversation_history: list[str] = []  # max 10 entries
current_utterance = ""

# Text tokens accumulate into current_utterance
# On sentence boundaries (. ? !), saved to history
# Last 6 entries passed to DeepSeek as context
```

A significant limitation: we only have the *agent's* text stream. The user's audio is not transcribed server-side (that would require a separate ASR pipeline), so the conversation history is one-sided. We pass the agent's filler text as a proxy for the user's intent, which works poorly for follow-up questions.

### 3.6 DeepSeek Integration

The DeepSeek call is straightforward:

```python
async def call_deepseek(self, query, conversation_context=""):
    messages = [
        {"role": "system", "content": f"""
            Answer questions about Transgate concisely (2-3 sentences).
            No markdown. Suitable for spoken delivery.
            {TRANSGATE_CONTEXT}
        """},
        {"role": "user", "content": f"Context: {conversation_context}\nQuestion: {query}"}
    ]

    response = await aiohttp_session.post(
        "https://api.deepseek.com/chat/completions",
        json={"model": "deepseek-chat", "max_tokens": 150, "temperature": 0.7, ...}
    )
    # ... sanitize response (strip markdown) for TTS
```

The response is sanitized to remove markdown formatting (headers, bold, links, bullet points) since the text will be vocalized, not displayed.

### 3.7 Deployment

The modified PersonaPlex server runs on Modal (serverless GPU):

- **GPU**: NVIDIA A10G (24GB VRAM), ~$0.46/hr
- **Container**: One conversation per container (stateful LM)
- **Scale-down**: 5-minute idle timeout
- **Cold start**: ~60s (model loading), ~50ms warm
- **Model**: `nvidia/personaplex-7b-v1` from HuggingFace (gated, ~15GB)

## 4. What Worked

### 4.1 TTS Injection

The TTS injection mechanism works correctly. Forcing text tokens through `LMGen.step()` produces natural-sounding speech that matches the established voice. The depformer generates appropriate prosody for the forced text, and the audio quality is indistinguishable from PersonaPlex's native generation.

Key implementation detail: feeding silence frames (`torch.zeros`) on the user input channel during injection, rather than sine wave tokens, produces cleaner output. The model interprets this as "the user is not speaking," which is the desired state during an injected response.

### 4.2 Protocol Extension

The `0x07` message type integrates cleanly with the existing protocol. The non-blocking queue in `opus_loop()` allows injection without disrupting the main audio pipeline. The client receives injected audio and text through the same `0x01`/`0x02` channels as native output, requiring no client-side changes.

### 4.3 Audio Pipeline

The browser-side audio pipeline (Opus capture → WebSocket → server → Mimi → LM → Mimi → Opus → WebSocket → AudioWorklet playback) achieves acceptable latency. Key parameters that matter:

- **Capture**: Opus at 24kHz, `RESTRICTED_LOWDELAY` application mode, 20ms frames, 2 frames per OGG page (~40ms chunks), complexity 0
- **Playback**: AudioWorklet (`MoshiProcessor`) with 80ms initial buffer, adaptive overflow/underrun handling, linear fade-in/fade-out to prevent clicks
- **Decoder warmup**: A synthetic OGG BOS page is sent before real audio to initialize the decoder's internal buffers, preventing dropped frames at conversation start

### 4.4 Modal Deployment

The serverless GPU deployment on Modal works well for a prototype. The `@modal.concurrent(max_inputs=1)` constraint correctly enforces one conversation per container (necessary because the LM is stateful). Auto-scaling handles concurrent users by spawning additional containers.

## 5. What Did Not Work

### 5.1 Instruction Following (The Core Failure)

PersonaPlex does not reliably follow system prompt instructions. This is the fundamental blocker.

**With the `!!!` marker**: The model produces `!!!` inconsistently. In testing, we observed:

- Sometimes the model outputs `!!!` as instructed
- Sometimes it answers the product question directly (incorrectly) without deferring
- Sometimes it produces `!` or `!!` instead of `!!!`
- Sometimes it ignores the routing rules entirely and generates generic conversational filler

This is not a prompt engineering failure that can be fixed with better wording. The Helium 7B backbone was trained for conversational dynamics using the Fisher English Corpus and synthetic customer service dialogues. It was not trained to follow structured instructions or produce specific output markers on command. The `<system>...<system>` prompt format (note: same tag at both ends, not XML-style closing tags) provides a weak form of instruction injection, but the model treats it as context rather than binding directives.

**With natural phrase detection**: Shifting to phrases like "let me check" was more reliable in the sense that PersonaPlex naturally produces filler phrases. However, the model does not produce them *selectively*. It says "let me check" for questions it could handle, and fails to say it for questions it cannot. The phrase is a conversational habit, not a routing decision. We also observed false positives where the model says "let me see" as a thinking pause during normal conversation, not as a defer signal.

### 5.2 The Controllability Problem

This is a deeper issue than just our use case. The Moshi/PersonaPlex architecture generates text tokens via the "Inner Monologue" mechanism, where text is a *prefix* to audio generation. The text tokens are sampled with `temp=0.7, top_k=25` -- relatively unconstrained. The model was trained to maximize conversational naturalness across its training distribution, not to follow output-format constraints.

Contrast this with instruction-tuned text LLMs, which are explicitly trained (via RLHF, DPO, or SFT on instruction-following data) to produce structured outputs on demand. PersonaPlex has no equivalent training signal. Asking it to reliably produce `!!!` is like asking a pre-instruction-tuning GPT-2 to reliably produce JSON -- it might do it sometimes by chance, but there is no training pressure ensuring it.

### 5.3 One-Sided Conversation History

Because the server only has access to the agent's text stream (the Inner Monologue), it cannot reliably extract the user's actual question. The user's speech goes through Mimi encoding and directly influences the LM's generation, but the user's words are never transcribed server-side.

This means `extract_last_user_query()` often returns nothing useful. The workaround -- passing the agent's filler text to DeepSeek -- works for first questions ("What's your pricing?" → agent says "Great question about pricing!" → we extract "pricing") but fails for follow-ups ("Tell me more" → agent says "Sure!" → we have no idea what "more" refers to).

### 5.4 Latency Budget

Even when defer detection works, the routing pipeline adds significant latency:

| Stage | Latency |
|-------|---------|
| Defer detection (text accumulation) | ~200-500ms (depends on phrase length) |
| DeepSeek API call | ~500-2000ms |
| TTS injection (~15 tokens) | ~750ms |
| **Total** | **~1.5-3.5s** |

During this time, PersonaPlex's native generation continues -- it keeps talking. This creates a collision: the model generates its own (often incorrect) response while we are waiting for DeepSeek's (correct) response. We attempted to suppress native generation during routing, but this breaks the full-duplex loop (the model needs continuous input to maintain state).

### 5.5 Prompt Conflict

The architecture resulted in two competing system prompts:

1. The **client-side prompt** (sent via URL query parameter `text_prompt`): uses natural phrases ("Let me check that for you")
2. The **server-side prompt** (sent via METADATA message `0x04`): uses `!!!` markers

Both are injected into PersonaPlex on connection. They contradict each other, and neither is reliably followed. This is partly an engineering oversight (the migration from markers to phrases was incomplete), but it also reveals the fragility of prompt-based control for this model class.

## 6. Discussion

### 6.1 Why Runtime Routing Is Hard for Speech Models

The core difficulty is that full-duplex speech models are **always generating**. Unlike a text LLM that waits for a complete user message before responding, PersonaPlex produces audio tokens continuously at 12.5 Hz. There is no natural "decision point" where the model can pause and route.

This means any routing mechanism must either:

1. **Detect intent post-hoc** (after the model has already started responding) -- which is what we attempted, leading to the collision problem described above
2. **Pre-classify user input** before it reaches the model -- which would require a separate ASR + intent classifier running in parallel, adding latency and complexity
3. **Train the model** to emit routing tokens as part of its native generation -- which requires fine-tuning on routing-annotated conversation data

Option 3 is the most promising but was outside the scope of this prototype.

### 6.2 The Controllability-Naturalness Tradeoff

There appears to be a fundamental tension in the Moshi architecture between conversational naturalness and controllability. The model's strength -- fluid, human-like conversation with natural interruptions and backchannels -- comes from training on real and synthetic conversations without explicit structure. Adding structured behavior (like reliable marker production) would likely require either:

- **Fine-tuning on marker-annotated data**: Training conversations where the agent correctly uses `!!!` in context, so the model learns this as a natural conversational pattern rather than an arbitrary instruction
- **Constrained decoding**: Modifying the text sampling to force specific tokens under certain conditions, similar to grammar-constrained generation in text LLMs. This is technically feasible (we already force tokens during TTS injection) but requires a reliable trigger condition

### 6.3 What About a Parallel ASR Pipeline?

One approach we considered but did not implement: running Whisper or a similar ASR model on the user's audio stream in parallel with PersonaPlex. This would give us:

- Actual user transcripts (solving the one-sided history problem)
- Intent classification on user queries (solving the trigger reliability problem)
- The ability to pre-route before PersonaPlex responds (solving the collision problem)

The cost is additional GPU memory and ~300ms latency for ASR. Given that PersonaPlex already uses ~20GB of the A10G's 24GB VRAM, this would likely require a larger GPU (A100) or a separate ASR service. This remains a viable direction.

## 7. Open Directions

### 7.1 Fine-Tuning for Routing Awareness

The most direct solution to the controllability problem is to fine-tune PersonaPlex on conversations that include routing behavior. The training data would include:

- Conversations where the agent correctly defers product questions with a specific token or phrase
- Conversations where the agent correctly handles simple queries directly
- The deferred responses (from DeepSeek) injected as ground truth continuations

This is a LoRA or full fine-tune on the Helium 7B backbone, estimated at 1-2 GPU-days on an A100. The PersonaPlex team has demonstrated that fine-tuning on synthetic conversations (customer service roles) works; routing is a natural extension.

However, fine-tuning addresses the *detection* problem but not the *timing* problem. Even a model that perfectly recognizes "this is a question I should defer" still faces the fundamental challenge that full-duplex generation is continuous. By the time the model emits a defer signal, it has already begun producing audio for a response. The model has no native concept of "pause and wait for external input" because the architecture was designed for uninterrupted streaming.

There is also a risk of capability degradation. PersonaPlex was fine-tuned from Moshi on synthetic conversations optimized for natural conversational dynamics. Adding routing behavior means training the model to do something orthogonal to its original objective. The DiVA work (Held et al., 2024) documents exactly this kind of capability forgetting during speech model fine-tuning, where new behaviors erode existing strengths.

The more interesting direction may be architectural rather than training-based: a lightweight routing head that runs ahead of audio generation (see Section 7.2), or a learned "hold" behavior where the model produces natural filler ("let me look into that", backchannels) while an external system prepares the response. This would treat routing not as a discrete decision point but as a conversational behavior the model can sustain over multiple frames, which fits the full-duplex paradigm better than a binary defer/don't-defer signal.

### 7.2 Embedding-Space Routing

Instead of detecting routing signals in the text token stream, we could examine the transformer's hidden states directly. The main transformer produces 4096-dimensional embeddings at each step. A lightweight classifier (linear probe or small MLP) trained on these embeddings could predict whether the current context requires external knowledge, potentially with higher accuracy than text-token pattern matching.

This has the advantage of making routing decisions at the representation level rather than the output level, and could trigger before the model commits to a response direction.

### 7.3 Dual-Model Streaming

A more ambitious architecture: run PersonaPlex and a text LLM simultaneously. PersonaPlex handles the voice stream. A parallel ASR + text LLM pipeline processes the same user audio and generates text responses. A mixer decides, on a per-turn basis, whether to use PersonaPlex's native response or inject the text LLM's response via TTS.

This is essentially the FunAudioLLM approach (An et al., 2024) but with PersonaPlex providing the full-duplex voice layer instead of a separate TTS model. The key challenge is the mixer's decision logic and the seamless blending of native and injected audio.

### 7.4 Constrained Decoding with Audio Awareness

A technically interesting direction: modify the text token sampling in `LMGen.step()` to bias toward routing tokens when the audio input patterns suggest a question. The Moshi architecture already cross-attends between text and audio streams. It may be possible to extract a "question probability" from the attention patterns and use it to bias text sampling toward defer tokens, without any fine-tuning.

### 7.5 Retrieval-Augmented Generation for Speech

RAG for text LLMs is well-established. Adapting it for speech models would involve:

1. Maintaining a vector store of product knowledge
2. Using the Inner Monologue text stream as a query
3. Retrieving relevant context and injecting it into the model's input

The challenge is that Moshi's context window is 3,000 tokens and the input channels are highly structured (interleaved codebooks with specific delay patterns). Injecting retrieved context would require careful integration with the existing prompt injection mechanism.

## 8. Reproducing This Work

### 8.1 Prerequisites

- NVIDIA GPU with 24GB+ VRAM (A10G, RTX 4090, or better)
- Python 3.10+, `libopus-dev`
- HuggingFace account with approved access to `nvidia/personaplex-7b-v1`

### 8.2 Server Setup

```bash
pip install moshi/.
export HF_TOKEN=<your_token>

# Basic server (no routing)
python -m moshi.server --ssl $(mktemp -d)

# With Modal deployment
pip install modal
modal secret create huggingface-secret HF_TOKEN=<your_token>
modal secret create deepseek-secret DEEPSEEK_API_KEY=<your_key>
modal serve modal_app.py
```

### 8.3 Key Files

| File | Description |
|------|-------------|
| `moshi/moshi/server.py` | WebSocket server with TTS injection and defer detection |
| `moshi/moshi/models/lm.py` | `LMGen` class with `step()` method for token forcing |
| `client/src/protocol/types.ts` | Protocol type definitions including `0x07` |
| `client/src/protocol/encoder.ts` | Message encoding/decoding |
| `modal_app.py` | Modal serverless deployment configuration |

### 8.4 Testing TTS Injection

The simplest way to verify TTS injection works:

```javascript
const ws = new WebSocket('wss://your-server/api');
ws.binaryType = 'arraybuffer';
ws.onopen = () => {
  const text = "Hello, this is an injected message.";
  const bytes = new TextEncoder().encode(text);
  const msg = new Uint8Array(1 + bytes.length);
  msg[0] = 0x07;
  msg.set(bytes, 1);
  ws.send(msg);
};
```

You should hear PersonaPlex speak the injected text in its current voice.

## 9. Conclusion

We set out to augment a full-duplex speech model with external LLM knowledge through runtime routing. The TTS injection mechanism works: we can force text through PersonaPlex's generation pipeline and produce natural-sounding speech. The routing mechanism does not work reliably: the model cannot be controlled via prompts to consistently signal when it needs help.

The fundamental issue is an impedance mismatch. Full-duplex speech models are trained for continuous, natural conversation. Routing requires discrete, reliable decision points. These goals conflict at the architectural level, and prompt engineering cannot bridge the gap.

Fine-tuning for routing awareness is the obvious next step, and would likely improve defer detection reliability. But it does not resolve the deeper tension: full-duplex models generate continuously, and routing requires discrete decision points. Even perfect detection leaves the timing problem unsolved. The more promising direction may be architectural: a routing head that operates ahead of audio generation, or a learned "hold" behavior where the model sustains natural filler while an external system prepares a response. This reframes routing as a conversational behavior rather than a binary decision, which fits the full-duplex paradigm better.

This work demonstrates both the potential and the current limits of hybrid speech architectures. The pieces exist (TTS injection, external LLM consultation, audio pipeline engineering), but the glue -- reliable, well-timed routing -- requires solutions at the architecture level that are not yet available in open-source full-duplex speech models. Until speech models are designed with external knowledge grounding as a first-class capability, analogous to how text LLMs learned function calling, runtime augmentation will remain fragile.

## References

- Defossez, A. et al. (2024). "Moshi: a speech-text foundation model for real-time dialogue." arXiv:2410.00037
- Roy, R. et al. (2026). "PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models." NVIDIA Research.
- Geng, Y. et al. (2025). "Control Illusion: The Failure of Instruction Hierarchies in Large Language Models." AAAI-26. arXiv:2502.15851
- Held, W. et al. (2024). "Distilling an End-to-End Voice Assistant Without Instruction Training Data." arXiv:2410.02678
- An, K. et al. (2024). "FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs." arXiv:2407.04051

## License

This document and the associated code modifications are provided under the MIT license, consistent with the PersonaPlex codebase. The PersonaPlex model weights are released under the NVIDIA Open Model License.
