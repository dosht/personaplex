# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)

# Phrases that trigger routing to DeepSeek
# These are natural filler phrases that Moshi reliably produces
DEFER_PHRASES = [
    "let me check",
    "let me find",
    "let me look",
    "let me see",
]
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    # Transgate product context for DeepSeek
    TRANSGATE_CONTEXT = """
# Transgate - AI Audio Transcription Platform

## Pricing Plans

### Pay As You Go - $1.49/hour
- Pay only for what you use
- Hours valid for 1 year
- Best for occasional users

### Premium - $14/month
- 20 hours of transcription per month
- Yearly option: $10/month

### Business - $21/month
- 40 hours of transcription per month
- Yearly option: $15/month

## Key Features
- 50+ languages supported
- 95-98% accuracy in clean audio
- AI Summarization and Smart Highlights
- Interactive AI Chat with transcripts
- No limit on file length
- Supports: mp3, wav, mp4, mov, avi, and all audio/video types
- REST API for developers
- HIPAA/GDPR compliant

## Free Trial
- 20 minutes free transcription
- No credit card required
"""

    async def call_deepseek(self, query: str, conversation_context: str = "") -> str:
        """
        Call DeepSeek API for complex product questions.
        Used when PersonaPlex defers with phrases like "let me check".

        Args:
            query: The recent conversation context indicating what user asked about
            conversation_context: Recent conversation history for additional context
        """
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        if not api_key:
            return "I apologize, but I'm unable to look that up right now. Please try again later."

        system_prompt = f"""You are a helpful voice assistant for Transgate, an AI-powered transcription platform.

{self.TRANSGATE_CONTEXT}

INSTRUCTIONS:
- Provide accurate, concise information about Transgate
- Keep responses brief (2-3 sentences max) as they will be spoken aloud
- Be friendly and conversational
- If asked about pricing, mention the free trial first
- Don't use bullet points or lists - speak naturally"""

        # Build the user message with context
        user_message = query
        if conversation_context:
            user_message = f"Recent conversation:\n{conversation_context}\n\nRespond to: {query}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 150,  # Shorter for voice
            "temperature": 0.7
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.deepseek.com/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        error_text = await resp.text()
                        logger.error(f"DeepSeek API error: {resp.status} - {error_text}")
                        return "I'm having trouble looking that up. Let me help you with something else."
        except asyncio.TimeoutError:
            logger.error("DeepSeek API timeout")
            return "That's taking longer than expected. Let me help you with something else."
        except Exception as e:
            logger.error(f"DeepSeek API exception: {e}")
            return "I encountered an issue looking that up. Please try again."

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        # Construct full voice prompt path (query params now optional)
        requested_voice_prompt_path = None
        voice_prompt_path = None
        voice_prompt_filename = request.query.get("voice_prompt", "")  # Optional
        if self.voice_prompt_dir is not None and voice_prompt_filename:
            requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, log warning but continue
            if not os.path.exists(requested_voice_prompt_path):
                clog.log("warning", f"Voice prompt '{voice_prompt_filename}' not found, using default")
                requested_voice_prompt_path = None
            else:
                voice_prompt_path = requested_voice_prompt_path

        if voice_prompt_path and self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)

        # Text prompt is optional - use empty list (not None) to avoid iteration errors
        text_prompt = request.query.get("text_prompt", "")
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(text_prompt)) if text_prompt else []
        seed = int(request["seed"]) if "seed" in request.query else None

        # TTS injection queue with backpressure (max 10 pending injections)
        tts_inject_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10)

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        try:
                            opus_reader.append_bytes(payload)
                        except ValueError as e:
                            # Stream may be closed, break out gracefully
                            clog.log("warning", f"opus_reader closed: {e}")
                            break
                    elif kind == 7:  # TTS inject
                        try:
                            text = message[1:].decode('utf-8')
                            if text.strip():  # Ignore empty strings
                                try:
                                    tts_inject_queue.put_nowait(text)
                                    clog.log("info", f"TTS inject queued: {text[:50]}...")
                                except asyncio.QueueFull:
                                    clog.log("error", "TTS inject queue full, rejecting request")
                                    error_msg = b"\x05" + bytes("TTS queue full", encoding="utf8")
                                    await ws.send_bytes(error_msg)
                            else:
                                clog.log("warning", "TTS inject received empty text")
                        except UnicodeDecodeError as e:
                            clog.log("error", f"TTS inject failed to decode UTF-8: {e}")
                            error_msg = b"\x05" + bytes(f"Invalid UTF-8: {e}", encoding="utf8")
                            await ws.send_bytes(error_msg)
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def process_tts_inject(text: str):
            """
            Convert text to speech using the LM's current voice state.

            This method:
            1. Tokenizes the input text
            2. Forces each text token through the LM to generate audio tokens
            3. Decodes audio tokens to PCM using Mimi
            4. Sends audio to client via opus_writer
            5. Sends transcript text back to client
            """
            try:
                clog.log("info", f"Processing TTS inject: {text[:50]}...")

                # Tokenize the input text
                tokens = self.text_tokenizer.encode(text)

                # Generate audio for each text token
                for text_token in tokens:
                    if close:
                        return

                    # Create a silence frame as user input (no user speaking during injection)
                    # Use zeros for silence - same approach as warmup
                    silence_chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
                    silence_codes = self.mimi.encode(silence_chunk)
                    _ = self.other_mimi.encode(silence_chunk)

                    # Force the text token through the LM
                    for c in range(silence_codes.shape[-1]):
                        audio_tokens = self.lm_gen.step(
                            input_tokens=silence_codes[:, :, c: c + 1],
                            text_token=text_token
                        )

                        if audio_tokens is None:
                            continue

                        # Decode audio tokens to PCM (24kHz mono)
                        assert audio_tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(audio_tokens[:, 1:9])
                        _ = self.other_mimi.decode(audio_tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()

                        # Append to opus writer (expects numpy array)
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())

                    # Send the text token back for transcript
                    # Replace sentence piece marker with space
                    _text = self.text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")

                    # Send as text message (0x02)
                    msg = b"\x02" + bytes(_text, encoding="utf8")
                    await ws.send_bytes(msg)

                    # Small yield to allow other async tasks to run
                    await asyncio.sleep(0.001)

                clog.log("info", f"TTS inject completed: {text[:50]}...")

            except Exception as e:
                clog.log("error", f"TTS injection failed: {e}")
                # Send error message to client
                error_msg = b"\x05" + bytes(f"TTS injection error: {str(e)}", encoding="utf8")
                await ws.send_bytes(error_msg)

        async def opus_loop():
            all_pcm_data = None

            # Conversation tracking for DeepSeek context
            conversation_history: list[str] = []  # Recent utterances
            current_utterance = ""  # Current sentence being built
            defer_in_progress = False

            async def handle_defer(filler_text: str, context: str):
                """Handle deferred query via DeepSeek API."""
                nonlocal defer_in_progress
                try:
                    clog.log("info", f"Defer detected. Filler: {filler_text[:50]}...")
                    clog.log("info", f"Context: {context[:100]}...")

                    # Call DeepSeek with filler as query hint and conversation as context
                    response = await self.call_deepseek(filler_text, context)
                    clog.log("info", f"DeepSeek response: {response[:50]}...")

                    # Inject response directly (internal TTS)
                    await process_tts_inject(response)

                    # Add response to conversation history
                    conversation_history.append(f"Assistant: {response}")
                    # Keep history manageable (last 5 exchanges)
                    while len(conversation_history) > 10:
                        conversation_history.pop(0)

                except Exception as e:
                    clog.log("error", f"Defer handling failed: {e}")
                finally:
                    defer_in_progress = False

            while True:
                if close:
                    return

                # Check for TTS injection requests (non-blocking)
                try:
                    inject_text = tts_inject_queue.get_nowait()
                    await process_tts_inject(inject_text)
                except asyncio.QueueEmpty:
                    pass

                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm is None or pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")

                            # Build current utterance
                            current_utterance += _text

                            # Check for defer phrases (natural filler phrases that trigger DeepSeek)
                            current_lower = current_utterance.lower()
                            detected_phrase = None
                            for phrase in DEFER_PHRASES:
                                if phrase in current_lower:
                                    detected_phrase = phrase
                                    break

                            if detected_phrase and not defer_in_progress:
                                defer_in_progress = True

                                # Extract filler text (what PersonaPlex said including the phrase)
                                filler_text = current_utterance.strip()
                                clog.log("info", f"[ROUTING] '{detected_phrase}' DETECTED -> DeepSeek | Context: '{filler_text}'")

                                # Build conversation context from history
                                context = "\n".join(conversation_history[-6:])  # Last 3 exchanges

                                # Add current filler to history
                                if filler_text:
                                    conversation_history.append(f"Assistant: {filler_text}")

                                # Reset for next utterance
                                current_utterance = ""

                                # Trigger DeepSeek call asynchronously
                                if filler_text:
                                    asyncio.create_task(handle_defer(filler_text, context))

                                # Continue sending text to client (the filler phrase is natural speech)
                                # Don't skip - let the user hear "let me check that for you"

                            # On sentence boundaries, save to history and reset
                            if _text.rstrip().endswith(('.', '?', '!')):
                                # Only log direct responses (not deferred ones)
                                if current_utterance.strip() and not defer_in_progress:
                                    clog.log("info", f"[ROUTING] DIRECT response: '{current_utterance.strip()}'")
                                    conversation_history.append(f"Assistant: {current_utterance.strip()}")
                                    # Keep history manageable
                                    while len(conversation_history) > 10:
                                        conversation_history.pop(0)
                                current_utterance = ""

                            # Send text to client
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                        else:
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if text_prompt:
            clog.log("info", f"text prompt: {text_prompt}")
        if voice_prompt_filename:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
    )
    logger.info("warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
