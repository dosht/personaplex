---
story_id: TTS-102
epic_id: TTS-1
title: DeepSeek Smart Routing Integration
status: ready
priority: high
points: 5
created: 2026-01-26
updated: 2026-01-26
dependencies: ['TTS-101']
---

# TTS-102: DeepSeek Smart Routing Integration

## User Story

**As a** Transgate landing page visitor
**I want** the voice assistant to provide detailed product information
**So that** I can learn about Transgate's features and pricing through natural conversation

## Context

This story implements the server-side `!!!` defer detection and DeepSeek integration in `moshi/server.py`. When PersonaPlex generates a response ending with `!!!`, the server detects this marker, calls DeepSeek API for a detailed response, and injects it via TTS.

**Architecture Decision:** Option C (Modal Server Integration) was chosen after POC validation. All routing logic happens server-side for lowest latency.

**Prerequisite:** TTS-101 (TTS Injection) must be complete. ✅

## Architecture Overview

```
User speaks → PersonaPlex STT + LM → Response with "!!!"
                                            ↓
                                   Server detects "!!!"
                                            ↓
                                   Extract user query
                                            ↓
                                   Call DeepSeek API
                                            ↓
                                   TTS inject response (internal)
                                            ↓
                                   Audio sent to user
```

## Acceptance Criteria

### Defer Detection

- [ ] Server buffers text output from LM generation
- [ ] `!!!` marker detected in text output stream
- [ ] Detection happens before audio is sent to client
- [ ] `!!!` marker creates pause/emphasis but is NOT vocalized

### Query Extraction

- [ ] User's last question extracted from conversation context
- [ ] Context includes recent conversation history (last 5 turns)
- [ ] Handles follow-up questions ("Tell me more about that")

### DeepSeek Integration

- [ ] DeepSeek API client initialized with API key from environment
- [ ] Transgate product context included in every request
- [ ] 30 second timeout for API calls
- [ ] Error handling with fallback message
- [ ] Response sanitized (no markdown) for natural TTS

### TTS Response Injection

- [ ] DeepSeek response passed to `process_tts_inject()`
- [ ] Audio generated and sent to client seamlessly
- [ ] No perceptible delay between filler and response

### System Prompt

- [ ] System prompt injected on WebSocket connection
- [ ] Prompt instructs PersonaPlex to use `!!!` for product questions
- [ ] Prompt includes routing rules (handle directly vs defer)

## Technical Implementation

### 1. Add DeepSeek Client to server.py

```python
# At top of file
import httpx
import os

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_TIMEOUT = 30.0

# Transgate product context
TRANSGATE_CONTEXT = """
# Transgate - AI Audio Transcription Platform

## Pricing Plans
- Pay As You Go: $1.49/hour (valid 1 year)
- Premium: $14/month (20 hours) - was $40
- Business: $21/month (40 hours) - was $60

## Key Features
- 50+ languages supported
- 95-98% accuracy in clean audio
- AI Summarization, Smart Highlights, Interactive Chat
- Speaker diarization (Indonesian locale)
- REST API for developers
- HIPAA/GDPR compliant

## Free Trial
- 20 minutes free transcription
- No credit card required
- Full access to all features
"""

async def call_deepseek(user_query: str, conversation_context: str = "") -> str:
    """Call DeepSeek API with Transgate context."""
    if not DEEPSEEK_API_KEY:
        return "I'm having trouble accessing that information right now."

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant answering questions about Transgate. Be concise (2-3 sentences). No markdown.\n\n{TRANSGATE_CONTEXT}"
        },
        {
            "role": "user",
            "content": f"Context: {conversation_context}\n\nQuestion: {user_query}"
        }
    ]

    try:
        async with httpx.AsyncClient(timeout=DEEPSEEK_TIMEOUT) as client:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": messages,
                    "max_tokens": 200,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        clog.log("error", f"DeepSeek API error: {e}")
        return "I'm having trouble with that question. Could you try asking differently?"
```

### 2. Add Defer Detection in opus_loop()

```python
# In ServerState class
DEFER_MARKER = "!!!"

# Track conversation history
conversation_history: list[dict] = []
text_buffer: str = ""

async def check_defer_and_respond(self, text: str, opus_writer, ws):
    """Check for defer marker and call DeepSeek if needed."""
    self.text_buffer += text

    if self.DEFER_MARKER in self.text_buffer:
        # Clean the marker from buffer
        clean_text = self.text_buffer.replace(self.DEFER_MARKER, "").strip()
        self.text_buffer = ""

        # Extract user's last question
        user_query = self.extract_last_user_query()
        if user_query:
            # Get conversation context (last few turns)
            context = self.get_conversation_context()

            # Call DeepSeek
            clog.log("info", f"Defer detected. Calling DeepSeek for: {user_query}")
            response = await call_deepseek(user_query, context)

            # Inject response via TTS
            await self.process_tts_inject(response, opus_writer, ws)

            # Update conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

def extract_last_user_query(self) -> str:
    """Get the user's most recent question from history."""
    for turn in reversed(self.conversation_history):
        if turn["role"] == "user":
            return turn["content"]
    return ""

def get_conversation_context(self, max_turns: int = 5) -> str:
    """Get recent conversation as context string."""
    recent = self.conversation_history[-max_turns:]
    lines = []
    for turn in recent:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)
```

### 3. Integrate with Text Output Stream

In the existing text generation/output code, call `check_defer_and_respond`:

```python
# After generating text token and before sending 0x02 message
_text = self.text_tokenizer.id_to_piece(text_token)
_text = _text.replace("▁", " ")

# Check for defer marker
await self.check_defer_and_respond(_text, opus_writer, ws)

# Send text to client (marker will create pause but not vocalize)
msg = b"\x02" + bytes(_text, encoding="utf8")
await ws.send_bytes(msg)
```

### 4. System Prompt Injection

Add system prompt handling on connection init:

```python
PERSONAPLEX_SYSTEM_PROMPT = """
<system>
You are Transgate's voice assistant on the landing page.

ROUTING RULES:

1. HANDLE DIRECTLY for:
   - Greetings: "Hi", "Hello", "Hey"
   - Acknowledgments: "Yes", "No", "Okay"
   - Backchannels: "Uh-huh", "Mm-hmm"
   - Simple navigation: "Go back", "Repeat that"

2. DEFER by ending with !!! (three exclamation marks) for:
   - Product questions (pricing, features, integrations)
   - How-to questions
   - Comparisons
   - Anything requiring specific knowledge

The backend will detect !!! and provide the detailed answer.

EXAMPLES:

User: "Hi there!"
You: "Hey! Welcome to Transgate. What can I help you with?"

User: "What's your pricing?"
You: "Great question! Let me tell you about our plans...!!!"

User: "How accurate is it?"
You: "Our accuracy is really impressive, let me explain...!!!"

NEVER say "please wait" or "one moment" - always contextual fillers.
NEVER answer product questions directly - always defer with !!!
</system>
"""

# In connection handler, inject system prompt
async def handle_connection(self, ws, request):
    # ... existing init code ...

    # Inject system prompt via text_prompt
    # This is passed to the LM as initial context
    self.system_prompt = PERSONAPLEX_SYSTEM_PROMPT
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | Yes |
| `DEEPSEEK_TIMEOUT` | API timeout in seconds (default: 30) | No |

## Testing Plan

### Unit Tests

```python
async def test_defer_detection():
    """Test !!! marker is detected correctly."""
    # Generate text ending with !!!
    # Verify check_defer_and_respond triggers
    pass

async def test_query_extraction():
    """Test user query extraction from history."""
    # Add conversation turns
    # Verify correct query extracted
    pass

async def test_deepseek_integration():
    """Test DeepSeek API call with mocked response."""
    # Mock httpx client
    # Verify correct request format
    # Verify response handling
    pass

async def test_deepseek_timeout():
    """Test timeout handling."""
    # Mock slow response
    # Verify fallback message used
    pass
```

### Manual Testing on Modal

1. Start PersonaPlex on Modal
2. Connect via browser
3. Say "What's your pricing?"
4. Verify:
   - Filler response ("Great question...")
   - Pause at `!!!` (no vocalization)
   - DeepSeek response spoken naturally
5. Test error handling:
   - Disable DEEPSEEK_API_KEY
   - Verify fallback message

### Integration Testing

1. Full end-to-end with Transgate frontend
2. Test various product questions:
   - Pricing
   - Features
   - Languages
   - Accuracy
   - API access
3. Test conversation flow:
   - Greeting → Question → Follow-up

## Definition of Done

- [ ] `!!!` detection implemented and tested
- [ ] DeepSeek API integration working
- [ ] System prompt injected on connection
- [ ] Conversation context tracked
- [ ] Error handling with fallback
- [ ] Environment variables documented
- [ ] Manual testing on Modal complete
- [ ] Code reviewed

## Dependencies

- **TTS-101**: TTS Injection (0x07) - ✅ Complete
- **DeepSeek API Key**: Required in Modal environment
- **Modal Deployment**: Server running on Modal GPU

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DeepSeek API latency | User waits too long | 30s timeout + filler audio plays immediately |
| API key exposure | Security breach | Use Modal secrets, never log key |
| Rate limiting | Service interruption | Implement retry with backoff |
| Query extraction fails | Wrong response | Include conversation context for clarity |

## Related Files

- `moshi/moshi/server.py` - Main server implementation
- `moshi/moshi/models/lm.py` - LM generation (read-only reference)
- `/Users/mu/Business/Transgate/frontend/main/docs/PERSONAPLEX_SOLUTION.md` - Architecture doc

## References

- [DeepSeek API Documentation](https://platform.deepseek.com/api-docs)
- [TTS-101 Implementation](./TTS-101-server-side-tts-injection.md)
- [PERSONAPLEX_SOLUTION.md](../../../../../../Business/Transgate/frontend/main/docs/PERSONAPLEX_SOLUTION.md)

---

**Created:** January 26, 2026
**Author:** Product Manager (via Claude Code)
