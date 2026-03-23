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

## Memory Write Mechanism (MVP)

Long-term memory is not updated at every turn. Instead, the system selectively writes memory based on simple rules.

### Trigger Conditions
A new memory entry is created when:
- the user provides stable personal information  
- the user expresses a recurring emotional pattern  
- the user mentions an important future event  
- the user expresses a clear preference  

### Extraction Strategy
In the MVP, memory is extracted using:
- rule-based heuristics
- keyword matching (e.g., "I am", "I like", "I have")

The system then:
1. summarizes the information into a short "fact"
2. assigns a timestamp
3. stores it in long-term memory

## Memory Usage in System Pipeline

The memory module is integrated into the system workflow as follows:

1. User sends a message
2. Short-term memory is updated
3. State tracker updates internal state
4. Memory store retrieves relevant long-term memory
5. Prompt builder constructs the prompt using:
   - persona
   - state
   - memory
   - recent conversation
6. Model generates response
7. Memory write policy decides whether to store new memory

## Interaction with Other Modules

Memory interacts with other components in the system:

- With state_tracker:
  emotional signals may influence both state updates and memory creation

- With policy_selector:
  memory can influence response style (e.g., more supportive tone)

- With prompt_builder:
  memory is injected into the prompt to improve personalization

This ensures the system produces more consistent and context-aware responses.

## Limitations (MVP)

The current memory system has several limitations:

- retrieval is based only on recency
- no semantic similarity search
- no mechanism to resolve conflicting memories
- no advanced privacy filtering

These limitations are acceptable for the MVP and will be improved in future work.
