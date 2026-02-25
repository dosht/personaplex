# Epic TTS-1: TTS Injection Integration

## Overview

Add server-side TTS injection and smart routing capabilities to PersonaPlex. This enables hybrid AI architectures where PersonaPlex handles real-time conversation while external AI systems (like DeepSeek) provide domain-specific responses that PersonaPlex speaks naturally.

**Architecture Decision:** Option C (Modal Server Integration) was chosen after POC validation. All routing logic (`!!!` detection, DeepSeek calls, TTS injection) happens server-side in `server.py` for lowest latency.

## Business Value

This epic enables PersonaPlex to be used as an intelligent voice interface for product demos, preserving its core value (200ms response, full-duplex, natural interruptions) while allowing DeepSeek to provide detailed product information.

## Key Use Case

**Transgate Voice Assistant**: PersonaPlex handles simple queries directly (greetings, acknowledgments). For product questions, it defers by ending with `!!!`, which triggers server-side DeepSeek API call. The response is injected via TTS internally, maintaining conversational flow with minimal latency.

## Technical Scope

- ✅ Protocol message type: 0x07 (TTS Inject)
- ✅ Server-side TTS injection queue and processing
- ✅ Integration with existing LM generation pipeline
- ✅ Protocol extension in TypeScript client
- 📋 Server-side `!!!` defer detection
- 📋 DeepSeek API integration
- 📋 System prompt injection

## Success Criteria

- [x] PersonaPlex speaks injected text using its current voice (TTS-101)
- [x] Audio quality matches normal speech generation (TTS-101)
- [x] Injection doesn't break ongoing conversation state (TTS-101)
- [x] Protocol documented and client library updated (TTS-101)
- [ ] `!!!` marker detected and triggers DeepSeek (TTS-102)
- [ ] DeepSeek responses spoken seamlessly (TTS-102)

## Epic Stories

| Story | Title | Status |
|-------|-------|--------|
| [TTS-101](./TTS-101-server-side-tts-injection.md) | Server-Side TTS Injection | ✅ Done |
| [TTS-102](./TTS-102-deepseek-smart-routing.md) | DeepSeek Smart Routing | 📋 Ready |

## Timeline

**Estimated Effort:**
- TTS-101: 5 points (2-3 hours) - ✅ Complete
- TTS-102: 5 points (2-3 hours) - 📋 Ready

**Dependencies:**
- ✅ Access to RunPod/Modal GPU environment
- ✅ Understanding of PersonaPlex LM generation pipeline
- 📋 DeepSeek API key in Modal environment

## Related Documentation

- Solution Architecture: `/Users/mu/Business/Transgate/frontend/main/docs/PERSONAPLEX_SOLUTION.md`
- Frontend Story: `/Users/mu/Business/Transgate/frontend/main/docs/product/epics/LAND-1-landing-pages/LAND-102-personaplex-voice-assistant.md`
- PersonaPlex Server: `moshi/moshi/server.py`
- LM Generation: `moshi/moshi/models/lm.py`
- Protocol Types: `client/src/protocol/types.ts`
