# Memory Schema

## Overview

The companion uses a two-tier memory system to maintain context within and across conversations. Memory enables the companion to reference past interactions, recall user preferences, and build a coherent relationship over time.

## Short-Term Memory

### Purpose

Maintain context within the current conversation session.

### Storage Format

```json
{
  "type": "short_term",
  "entries": [
    {
      "turn": 3,
      "role": "user",
      "content": "I have a big exam tomorrow.",
      "emotion": "anxious"
    },
    {
      "turn": 4,
      "role": "assistant",
      "content": "That sounds stressful. What subject is it?"
    }
  ]
}
```

### Behavior

- Stores the last N turns of conversation (default: N = 10).
- Automatically trimmed when context exceeds the window.
- Used directly in prompt construction as conversation history.

## Long-Term Memory

### Purpose

Store persistent facts about the user that span across sessions.

### Storage Format

```json
{
  "type": "long_term",
  "entries": [
    {
      "fact": "User is a graduate student at CMU.",
      "source_turn": 5,
      "timestamp": "2026-03-20T14:30:00"
    },
    {
      "fact": "User tends to feel overwhelmed before deadlines.",
      "source_turn": 12,
      "timestamp": "2026-03-20T14:45:00"
    },
    {
      "fact": "User's favorite hobby is hiking.",
      "source_turn": 22,
      "timestamp": "2026-03-21T10:00:00"
    }
  ]
}
```

### What to Remember

- Personal facts (name, school, major, hobbies)
- Recurring emotional patterns ("user often feels stressed on Mondays")
- Preferences ("user prefers short responses")
- Important events ("user has a midterm on Friday")

### What NOT to Remember

- Exact conversation transcripts (that's short-term memory)
- Sensitive information the user asks to forget
- One-off filler messages with no informational content

## Memory Retrieval

### Current Approach (MVP)

- Return the most recent K long-term memories (default: K = 3).
- No semantic search — simple recency-based retrieval.

### Future Approach

- Use embedding-based similarity search (e.g., FAISS + sentence-transformers).
- Rank memories by relevance to the current user message.
- Combine recency and relevance scoring.

## Memory as Prompt Context

Retrieved memories are injected into the system prompt as:

```
Relevant memories:
- User is a graduate student at CMU.
- User tends to feel overwhelmed before deadlines.
- User has a midterm on Friday.
```

## Memory Persistence

- Short-term memory resets each session.
- Long-term memory persists across sessions (stored as JSON file for MVP).
- Future work: use a lightweight database or vector store.
