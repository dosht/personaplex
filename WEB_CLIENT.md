# Building a Real-Time Voice Client for PersonaPlex: WebSocket Binary Protocol, Opus WASM Codecs, and AudioWorklet Playback

## Overview

This document describes the web client implementation for PersonaPlex -- a full-duplex speech-to-speech model that generates audio at 12.5 Hz (80ms frames). Building a browser client for this system turned out to be a non-trivial audio engineering exercise. The model sends and receives Opus-encoded audio over a binary WebSocket protocol, and the client needs to capture microphone audio, encode it, send it with minimal latency, receive generated audio, decode it, and play it back -- all in real time, with adaptive jitter buffering to handle network variance.

We built two clients during this project:

1. **The native PersonaPlex client** (`client/`): A Vite + React + TypeScript application that ships with this repository. It provides a full conversation UI with model parameter controls, stereo recording, and audio visualization.
2. **An embeddable widget client** (in the [Transgate frontend repo](https://github.com/dosht/voice-agent-demo)): A Next.js React feature module (`src/features/voice-demo/`) designed as a landing page widget. It ports the native client's audio pipeline into a set of React hooks.

Both clients share the same audio architecture. This document covers the common patterns, the binary protocol, and the engineering challenges we worked through.

## 1. The Binary WebSocket Protocol

### 1.1 Message Framing

Every message is a single WebSocket binary frame:

```
[1 byte: message type] [N bytes: payload]
```

There are no length prefixes or delimiters -- WebSocket framing provides message boundaries. The type byte determines how to interpret the payload:

| Byte | Type | Direction | Payload Encoding |
|------|------|-----------|------------------|
| `0x00` | Handshake | Server -> Client | Version byte + model byte (or empty) |
| `0x01` | Audio | Bidirectional | Raw OGG Opus page bytes |
| `0x02` | Text | Server -> Client | UTF-8 string (SentencePiece tokens) |
| `0x03` | Control | Client -> Server | 1 byte: `0x00`=start, `0x01`=endTurn, `0x02`=pause, `0x03`=restart |
| `0x04` | Metadata | Bidirectional | UTF-8 JSON string |
| `0x05` | Error | Server -> Client | UTF-8 error string |
| `0x06` | Ping | Bidirectional | Echo-back (return the exact bytes) |
| `0x07` | TTS Inject | Client -> Server | UTF-8 text to be synthesized |

The `0x07` TTS Inject type was added by this fork for the smart routing experiment (see [SMART_ROUTING.md](./SMART_ROUTING.md)).

### 1.2 Encoding and Decoding

The encoding is straightforward byte concatenation:

```typescript
function encodeMessage(message: WSMessage): Uint8Array {
  switch (message.type) {
    case "audio":
      const audio = new Uint8Array(1 + message.data.length);
      audio[0] = 0x01;
      audio.set(message.data, 1);
      return audio;

    case "text":
      const textBytes = new TextEncoder().encode(message.data);
      const text = new Uint8Array(1 + textBytes.length);
      text[0] = 0x02;
      text.set(textBytes, 1);
      return text;

    case "tts_inject":
      const injectBytes = new TextEncoder().encode(message.data);
      const inject = new Uint8Array(1 + injectBytes.length);
      inject[0] = 0x07;
      inject.set(injectBytes, 1);
      return inject;

    // ... other types
  }
}
```

Decoding mirrors this: slice byte 0 as the type, interpret the rest based on the type. Audio payloads are kept as raw `Uint8Array`; text payloads are decoded with `TextDecoder`.

### 1.3 Connection Lifecycle

```
1. Client opens WebSocket to wss://{host}/api/chat?{params}
2. ws.binaryType = "arraybuffer"
3. On ws.open: status = "connecting" (NOT "connected" yet)
4. Server processes voice prompt + system prompt injection (~2-5s)
5. Server sends Handshake (0x00)
6. Client receives 0x00: status = "connected", audio pipeline starts
7. Bidirectional audio/text streaming begins
8. On disconnect or 10s inactivity: connection closes
```

The key detail is that WebSocket `open` does not mean the server is ready. The server needs several seconds to load the voice prompt embeddings and process the system prompt through the LM. Only after the Handshake message should the client start sending audio.

### 1.4 Query Parameters

The WebSocket URL carries model configuration:

```
wss://host/api/chat?
  text_temperature=0.7      # Inner Monologue sampling temperature
  text_topk=25               # Text token top-k
  audio_temperature=0.8      # Audio codebook sampling temperature
  audio_topk=250             # Audio token top-k
  pad_mult=0                 # Padding multiplier
  repetition_penalty=1.0     # Token repetition penalty
  repetition_penalty_context=64
  text_prompt=<system prompt> # System prompt text (URL-encoded)
  voice_prompt=NATF0.pt      # Voice embedding file
```

## 2. Audio Capture: Microphone to Opus

### 2.1 The Problem

The server expects OGG Opus pages at 24kHz. Browsers capture audio at their native sample rate (typically 44.1kHz or 48kHz). We need to:

1. Capture microphone audio
2. Resample to 24kHz
3. Encode with Opus codec
4. Package into OGG pages
5. Send over WebSocket with minimal latency

We use the [opus-recorder](https://github.com/nicklasnygren/opus-recorder) library, which bundles a WASM-compiled Opus encoder inside a Web Worker.

### 2.2 Encoder Configuration

```typescript
const recorder = new Recorder({
  sourceNode: audioContext.createMediaStreamSource(micStream),
  encoderPath: '/opus-recorder/encoderWorker.min.js',

  // Buffer size: scales PersonaPlex's base of 960 samples at 24kHz
  // to match the actual AudioContext sample rate
  bufferLength: Math.round(960 * audioContext.sampleRate / 24000),

  encoderSampleRate: 24000,    // Output: 24kHz (PersonaPlex native rate)
  encoderFrameSize: 20,        // 20ms Opus frames (standard for voice)
  encoderApplication: 2049,    // OPUS_APPLICATION_RESTRICTED_LOWDELAY
  encoderComplexity: 0,        // Fastest encoding (0-10)
  numberOfChannels: 1,         // Mono
  recordingGain: 1,            // Unity gain

  streamPages: true,           // Emit OGG pages as they're ready
  maxFramesPerPage: 2,         // 2 frames per page = ~40ms per page
  resampleQuality: 3,          // libspeexdsp quality (0-10)
});
```

The parameter choices deserve explanation:

**`encoderApplication: 2049` (RESTRICTED_LOWDELAY)**: Opus has three application modes: `2048` (VOIP), `2049` (Restricted Low Delay), and `2051` (Audio). We initially used VOIP mode but switched to RESTRICTED_LOWDELAY after observing latency issues. RESTRICTED_LOWDELAY minimizes algorithmic delay at the cost of some compression efficiency -- a worthwhile tradeoff for real-time conversation. This was one of our iteration discoveries; the PersonaPlex native client also uses this mode.

**`maxFramesPerPage: 2`**: Each OGG page contains 2 Opus frames (2 x 20ms = 40ms). Fewer frames per page means smaller, more frequent network transmissions. A single frame per page would reduce latency further but increases framing overhead. Two frames is the sweet spot we settled on.

**`encoderComplexity: 0`**: The lowest complexity setting. For voice at low bitrate, higher complexity barely improves quality but increases CPU. In a real-time system where the encoding thread competes with decoding and the main UI thread, saving CPU cycles matters.

**`streamPages: true`**: This is critical. Without it, opus-recorder buffers the entire OGG file until `stop()` is called. With `streamPages`, each OGG page is emitted via `ondataavailable` as soon as it's encoded, enabling real-time streaming.

### 2.3 Data Flow

```
Microphone
  -> getUserMedia({ audio: {
       channelCount: 1,
       sampleRate: { ideal: 24000 },
       echoCancellation: true,
       noiseSuppression: true,
       autoGainControl: true
     }})
  -> AudioContext.createMediaStreamSource(stream)
  -> opus-recorder Web Worker
     [Resample: browser rate -> 24kHz]
     [Encode: Opus RESTRICTED_LOWDELAY, 20ms frames, complexity 0]
     [Package: OGG pages, 2 frames/page]
  -> ondataavailable(ArrayBuffer)
  -> prepend 0x01 type byte
  -> WebSocket.send(Uint8Array)
```

The server-side Opus decoder is `sphn.OpusStreamReader` (from Kyutai's `sphn` Python audio library), which accepts OGG Opus page streams directly.

## 3. Audio Playback: Opus to Speakers

This is where the real engineering challenge lives. The server generates audio tokens at 12.5 Hz and sends Opus-encoded frames over the WebSocket. The client needs to decode them and play them back smoothly despite network jitter, without introducing perceptible latency.

### 3.1 The Three Attempts

We went through three iterations to get playback right:

**Attempt 1: Naive AudioBufferSourceNode scheduling.** Create an `AudioBuffer` for each decoded frame, create an `AudioBufferSourceNode`, connect it to the destination, and schedule it with `source.start(startTime)`:

```javascript
// DON'T DO THIS for real-time speech
const buffer = ctx.createBuffer(1, samples.length, ctx.sampleRate);
buffer.getChannelData(0).set(samples);
const source = ctx.createBufferSource();
source.buffer = buffer;
source.connect(ctx.destination);
source.start(Math.max(scheduledTime, ctx.currentTime));
scheduledTime += buffer.duration;
```

This approach fails in practice: scheduling drift accumulates over time, there's no way to handle buffer underruns gracefully (you get silence gaps or clicks), and there's no mechanism to drop old packets when the network delivers a burst of delayed audio. We observed 50-100ms+ of accumulated drift after a few minutes of conversation.

**Attempt 2: Decode in the main thread with basic queuing.** Slightly better, but the main thread processing introduces jank during React renders, and the lack of adaptive buffering still produces audible glitches.

**Attempt 3: AudioWorklet + adaptive jitter buffer.** This is what the PersonaPlex native client uses, and what we eventually ported. It runs in a dedicated audio rendering thread with microsecond-precision scheduling.

### 3.2 The Playback Pipeline

```
WebSocket message (type 0x01)
  -> strip type byte, extract Opus payload
  -> postMessage to Decoder Worker (with Transferable ArrayBuffer for zero-copy)
  -> WASM Opus decoder (libopus compiled to WebAssembly)
  -> Float32Array(960) PCM frames at 24kHz
  -> postMessage back to main thread
  -> worklet.port.postMessage({ type: "audio", frame: Float32Array })
  -> MoshiProcessor (AudioWorkletProcessor) adaptive jitter buffer
  -> AudioContext.destination (speakers)
```

Three separate threads are involved:

1. **Main thread**: Receives WebSocket messages, dispatches to decoder worker, receives decoded audio, forwards to worklet
2. **Decoder Worker thread**: Runs WASM-compiled libopus, decodes OGG pages to PCM
3. **Audio rendering thread**: Runs the AudioWorklet processor, outputs audio at the hardware sample rate

### 3.3 Opus Decoder: WASM in a Web Worker

The decoder uses `decoderWorker.min.js` + `decoderWorker.min.wasm` from the opus-recorder library. The WASM binary (150KB) contains a compiled libopus decoder with Speex resampler.

**Initialization:**

```typescript
const worker = new Worker('/opus-recorder/decoderWorker.min.js');
worker.postMessage({
  command: 'init',
  bufferLength: Math.round(960 * audioContext.sampleRate / 24000),
  decoderSampleRate: 24000,
  outputBufferSampleRate: audioContext.sampleRate,
  resampleQuality: 0,   // fastest resampling
});
```

When both rates are 24kHz (which they are if we create the AudioContext at 24kHz), `bufferLength` is 960 samples and no resampling occurs. The decoder outputs `Float32Array(960)` per frame -- 40ms of audio at 24kHz.

**Decoding:**

```typescript
// Send OGG pages to decoder (zero-copy via Transferable)
worker.postMessage(
  { command: 'decode', pages: audioData },
  [audioData.buffer]  // transfer ownership, avoid memory copy
);

// Receive decoded PCM
worker.onmessage = (e) => {
  const samples: Float32Array = e.data[0];
  // Forward to AudioWorklet...
};
```

The Transferable buffer optimization matters. Each audio message is ~200-400 bytes of Opus data. Without transfer, the browser copies the ArrayBuffer when posting to the worker, creating garbage that the GC must collect. With transfer, ownership moves to the worker with zero copy.

### 3.4 The Decoder Warmup Problem

This was one of our more puzzling debugging sessions. On the first audio message from the server, the decoder would silently drop it. The second and subsequent messages decoded fine.

The root cause: the WASM Opus decoder requires an OGG stream initialization sequence before it can decode audio packets. The OGG format starts with a BOS (Beginning of Stream) page containing an OpusHead header that tells the decoder the channel count, sample rate, and pre-skip. The server's OGG stream starts with this header, but the decoder's WASM initialization (allocating internal buffers, creating the Opus decoder state) takes time. If the first real audio packet arrives before the WASM has finished processing the BOS page, it gets dropped.

**The fix: send a synthetic BOS page during pre-warming.** Before the WebSocket even connects, we initialize the decoder and feed it a hand-crafted 47-byte OGG BOS page:

```typescript
function createWarmupBosPage(): Uint8Array {
  const page = new Uint8Array(47);

  // OGG page header
  page.set([0x4F, 0x67, 0x67, 0x53]);  // "OggS" magic
  page[4] = 0x00;                        // version
  page[5] = 0x02;                        // BOS flag
  // bytes 6-13: granule position (0)
  page[14] = 0x01;                       // stream serial = 1
  // bytes 18-25: page sequence (0), CRC (0)
  page[26] = 0x01;                       // 1 segment
  page[27] = 0x13;                       // segment size: 19 bytes

  // OpusHead payload (19 bytes)
  page.set([
    0x4F, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64,  // "OpusHead"
    0x01,                                               // version 1
    0x01,                                               // 1 channel (mono)
    0x38, 0x01,                                         // pre-skip: 312 (LE)
    0x80, 0xBB, 0x00, 0x00,                            // sample rate: 48000 (LE)
    0x00, 0x00,                                         // gain: 0
    0x00                                                // mapping family: 0
  ], 28);

  return page;
}
```

Some details on the byte values:

- **Pre-skip of 312 samples**: This is the standard Opus encoder algorithmic delay. The decoder discards this many samples at the start of the stream to compensate for the encoder's lookahead. 312 samples at 48kHz = 6.5ms.
- **Sample rate of 48000**: This is the Opus *internal* sample rate. Opus always operates at 48kHz internally regardless of the configured output rate. The decoder downsamples to the requested output rate (24kHz in our case).
- **CRC of 0x00000000**: The opus-recorder decoder worker does not validate OGG CRC checksums, so we skip computing it. A production OGG implementation should compute the CRC32 per the OGG spec.

The warmup sequence:

```
1. User clicks "Start Call"
2. prewarmDecoder() called immediately:
   a. Create Worker
   b. Send init command (bufferLength, sampleRates)
   c. Wait 100ms for WASM to load
   d. Send warmup BOS page
   e. Wait 200ms for decoder to process
   f. Mark isDecoderReady = true
3. connect() opens WebSocket
4. Server processes voice/text prompts (~2-5s)
5. Server sends Handshake (0x00)
6. Audio streaming begins -- decoder is already warm
```

The audio callback is gated on the decoder readiness flag:

```typescript
onAudioReceived: isDecoderReady ? queueAudio : undefined
```

If `isDecoderReady` is false, audio messages are simply ignored (the callback is `undefined`). This eliminates any race conditions between decoder initialization and audio arrival.

### 3.5 The AudioWorklet: MoshiProcessor

The `MoshiProcessor` is an `AudioWorkletProcessor` that runs in the audio rendering thread. It implements an adaptive jitter buffer that absorbs network timing variance while maintaining low latency. This was ported from the PersonaPlex native client (`src/audio-processor.ts`).

**Core parameters (at 24kHz sample rate):**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `frameSize` | 1920 samples (80ms) | Reference unit, matches PersonaPlex frame rate |
| `initialBufferSamples` | 1920 (80ms) | Minimum buffered before playback starts |
| `partialBufferSamples` | 240 (10ms) | Additional delay after initial buffer fills |
| `maxBufferSamples` | 240 (10ms) | Max excess before dropping old packets |

All three "extra" parameters (partial, max) adapt upward by 120 samples (5ms) per event, capped at 1920 samples (80ms).

**The buffering algorithm:**

```
Incoming frames -> push to frames[] queue

State machine:
  NOT STARTED:
    if currentSamples() >= initialBufferSamples:
      started = true
      remainingPartialBufferSamples = partialBufferSamples

  STARTED, PARTIAL DELAY:
    output silence
    remainingPartialBufferSamples -= renderQuantumSize
    (wait for partial buffer to drain)

  PLAYING:
    copy samples from frames[] to output buffer
    track totalAudioPlayed, actualAudioPlayed

  OVERFLOW (buffer > initialBuffer + partialBuffer + maxBuffer):
    drop oldest frames until buffer = initialBuffer + partialBuffer
    maxBufferSamples += 5ms (adapt, up to 80ms cap)

  UNDERRUN (ran out of frames mid-render):
    apply fade-out to partial output
    partialBufferSamples += 5ms (adapt, up to 80ms cap)
    reset to NOT STARTED (rebuffer)
```

**Fade-in and fade-out prevent audible clicks:**

```javascript
// Fade-in: first output after buffering
for (let i = 0; i < outIdx; i++) {
  output[i] *= i / outIdx;   // linear ramp 0.0 -> 1.0
}

// Fade-out: underrun mid-render
for (let i = 0; i < outIdx; i++) {
  output[i] *= (outIdx - i) / outIdx;  // linear ramp 1.0 -> 0.0
}
```

This is a simple linear crossfade. More sophisticated (cosine, raised-cosine) fades would produce fewer spectral artifacts, but the linear fade is sufficient for the short durations involved (typically <10ms of fade).

**The render quantum**: The AudioWorklet's `process()` method is called with 128-sample buffers. At 24kHz, that's ~5.3ms per call. At 48kHz (some browsers), it's ~2.7ms. The processor must be fast enough to complete within this budget, or audio dropouts occur.

**Latency accounting:**

The worklet tracks two time metrics:
- `totalAudioPlayed`: wall-clock time since first output (including silence during underruns)
- `actualAudioPlayed`: actual non-silence PCM samples played

The difference `totalAudioPlayed - actualAudioPlayed` represents time spent in underrun silence. The worklet also computes `delay = micDuration - timeInStream`, which estimates the end-to-end latency between the user speaking and hearing the response.

### 3.6 Total Playback Latency Budget

```
Network transit:                  ~20-100ms (varies)
Opus decoding (WASM):             ~2-5ms
Worker -> Main -> Worklet:        ~1-3ms (postMessage overhead)
AudioWorklet initial buffer:      80ms (fixed)
AudioWorklet partial buffer:      10-80ms (adaptive)
Audio hardware output buffer:     ~5-10ms (OS-dependent)
                                  ─────────
Total:                            ~120-280ms
```

This sits within the acceptable range for real-time conversation. PersonaPlex's own generation latency is ~200ms, so the total user-perceived latency (speak -> hear response) is roughly 320-480ms.

## 4. Stereo Recording

The native client records a stereo WebM/MP4 with server audio on the left channel and user microphone on the right channel. This is useful for debugging and evaluation.

```
MoshiProcessor (AudioWorklet)
  |
  +--- connect(output 0, merger input 0) ---> ChannelMergerNode(2)
                                                    |
MediaStreamSource (mic)                             |
  |                                                 v
  +--- connect(output 0, merger input 1) ---> MediaStreamDestination
                                                    |
                                                    v
                                              MediaRecorder
                                              (webm 128kbps)
                                                    |
                                                    v
                                              Blob -> download
```

The `ChannelMergerNode` takes two mono inputs and produces a stereo stream. `MediaRecorder` captures this as a WebM (or MP4 on Safari) blob. WebM blobs have a known issue where the duration metadata is missing; the `webm-duration-fix` library patches this before creating the download URL.

## 5. Text Streaming

Text tokens arrive as `0x02` messages, each containing a SentencePiece token with the `\u2581` (lower one eighth block) marker replaced by a space on the server side. The client simply appends each token to a list:

```typescript
socket.addEventListener("message", (e) => {
  const message = decodeMessage(new Uint8Array(e.data));
  if (message.type === "text") {
    setText(prev => [...prev, message.data]);
  }
});
```

Text and audio messages are interleaved on the same WebSocket. There is no synchronization between them -- text tokens arrive roughly in sync with the corresponding audio because the server generates them in lockstep (text is a prefix to audio in the Moshi architecture), but the client makes no attempt to align them.

## 6. What We Learned

### Start with AudioWorklet, not AudioBufferSourceNode

Our first attempt at playback used `AudioBufferSourceNode` scheduling. This is the "obvious" Web Audio API approach, but it is fundamentally wrong for real-time streaming. `AudioBufferSourceNode` is designed for pre-loaded, known-duration audio. For streaming, you need an `AudioWorkletProcessor` that pulls from a buffer on the audio rendering thread. This is a common pitfall in Web Audio real-time applications -- the API surface suggests `AudioBufferSourceNode` for simple cases, but real-time streaming is not a simple case.

### RESTRICTED_LOWDELAY makes a real difference

Switching the Opus encoder from VOIP (2048) to RESTRICTED_LOWDELAY (2049) reduced perceivable end-to-end latency. The VOIP mode includes additional signal processing (forward error correction, loss concealment) that adds algorithmic delay. For a reliable WebSocket connection, these features are unnecessary.

### Pre-warm everything

The decoder WASM load, AudioWorklet module registration, and AudioContext creation all take time. If you do them lazily on first audio arrival, the user hears a silence gap or missed first words. Pre-warming the decoder (with the synthetic BOS page) during the WebSocket connection phase -- which itself has a multi-second wait while the server loads the voice prompt -- hides these latencies behind an already-existing wait.

### The browser is not a real-time audio environment

Between garbage collection pauses, React re-renders, and the main thread event loop, the browser's main thread is unreliable for audio timing. The AudioWorklet's rendering thread is the only reliable place to handle audio output. Minimize what happens on the main thread: decode in a Worker, play in a Worklet, use the main thread only for routing messages between them.

### Zero-copy matters at scale

Using `Transferable` objects in `postMessage` (passing `[arrayBuffer]` as the second argument) avoids copying audio data between threads. For a single conversation this is a micro-optimization, but when the system is streaming audio at 25 messages/second (12.5 Hz frame rate, each decoded into PCM), the cumulative allocation pressure is significant. We observed smoother playback after adding Transferable transfers.

## 7. File Reference

### Native Client (`client/`)

| File | Purpose |
|------|---------|
| `src/protocol/types.ts` | Message type definitions, control maps |
| `src/protocol/encoder.ts` | Binary encode/decode for all message types |
| `src/audio-processor.ts` | MoshiProcessor AudioWorklet (adaptive jitter buffer) |
| `src/decoder/decoderWorker.ts` | Decoder Worker factory, pre-warming, BOS page creation |
| `src/pages/Conversation/hooks/useSocket.ts` | WebSocket lifecycle, inactivity timeout |
| `src/pages/Conversation/hooks/useServerAudio.ts` | Server audio decoding pipeline |
| `src/pages/Conversation/hooks/useUserAudio.ts` | Mic capture + Opus encoding |
| `src/pages/Conversation/hooks/useServerText.ts` | Text token accumulation |
| `src/pages/Conversation/hooks/useModelParams.ts` | Model parameter state + presets |
| `src/pages/Conversation/Conversation.tsx` | Main orchestrator, stereo recording |
| `public/assets/decoderWorker.min.js` | WASM Opus decoder Worker (28KB) |
| `public/assets/decoderWorker.min.wasm` | Compiled libopus decoder (150KB) |

### Embeddable Widget Client (separate repo)

| File | Purpose |
|------|---------|
| `src/features/voice-demo/hooks/usePersonaPlex.ts` | WebSocket + binary protocol |
| `src/features/voice-demo/hooks/useAudioCapture.ts` | Mic + Opus encoding |
| `src/features/voice-demo/hooks/useAudioPlayback.ts` | Decoder + WorkletNode |
| `src/features/voice-demo/hooks/useCallTimer.ts` | Call duration countdown |
| `src/features/voice-demo/components/VoiceDemoWidget.tsx` | Modal UI widget |
| `public/audio-processor.js` | MoshiProcessor (ported from native) |
| `public/opus-recorder/encoderWorker.min.js` | WASM Opus encoder Worker |
| `public/opus-recorder/decoderWorker.min.js` | WASM Opus decoder Worker |
| `public/opus-recorder/decoderWorker.min.wasm` | Compiled libopus (150KB) |

## 8. Browser Compatibility

The client requires:

- **WebSocket binary frames**: All modern browsers
- **AudioContext** (or `webkitAudioContext`): Chrome 35+, Firefox 25+, Safari 14.1+
- **AudioWorklet**: Chrome 64+, Firefox 76+, Safari 14.1+
- **WebAssembly**: Chrome 57+, Firefox 52+, Safari 11+
- **getUserMedia**: Requires HTTPS (or localhost)
- **opus-recorder**: Uses `ScriptProcessorNode` internally for capture (deprecated but universally supported); encoding happens in the Worker

Safari support is the limiting factor. AudioWorklet was added in Safari 14.1 (April 2021). Older Safari versions would need a `ScriptProcessorNode` fallback for the playback path, which we have not implemented.

The native client generates self-signed HTTPS certificates for development (`SSL_DIR=$(mktemp -d)`) because `getUserMedia` requires a secure context. The Vite dev server is configured with `server.https: true` and binds to `0.0.0.0` for network access.
