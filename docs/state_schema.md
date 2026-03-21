# Relationship State Schema

## Overview

The companion maintains a set of relationship state variables that evolve over the course of a conversation. These variables influence how the companion responds — its tone, depth, and interaction strategy.

## Core State Variables

### 1. Affection (0–100)

- **Description**: How positively the companion feels about the user based on interaction history.
- **Default**: 50
- **Increases when**: User expresses gratitude, shares personal stories, engages warmly.
- **Decreases when**: User is hostile, dismissive, or rude.
- **Effect on behavior**: Higher affection → warmer tone, more personal responses.

### 2. Trust (0–100)

- **Description**: How much the user has opened up and how safe the interaction feels.
- **Default**: 50
- **Increases when**: User shares vulnerable feelings, responds honestly, continues conversation.
- **Decreases when**: User is aggressive, contradicts previous statements, pushes boundaries.
- **Effect on behavior**: Higher trust → companion shares more reflective or deeper responses.

### 3. Intimacy (0–100)

- **Description**: Depth of the relationship — how well the companion "knows" the user.
- **Default**: 50
- **Increases when**: Multiple sessions, user shares recurring topics, memories accumulate.
- **Decreases when**: Long periods of absence, user resets context.
- **Effect on behavior**: Higher intimacy → more references to past conversations, personalized responses.

### 4. Mood (categorical)

- **Description**: The companion's current emotional register, influenced by user input.
- **Default**: `neutral`
- **Possible values**: `neutral`, `concerned`, `cheerful`, `reflective`, `tense`
- **Updated by**: Emotion classification of user input.
- **Effect on behavior**: Directly shapes tone and word choice.

### 5. Energy (0–100)

- **Description**: Conversational energy level — how engaged or fatigued the interaction feels.
- **Default**: 70
- **Increases when**: Conversation is lively, user asks questions, humor is present.
- **Decreases when**: Conversation is repetitive, user gives short responses.
- **Effect on behavior**: Lower energy → shorter responses, more check-in questions.

## State Update Rules

State updates are applied **after each user message** and **before generating the reply**.

### Rule Format

```
IF <condition on user message or history>
THEN <adjust variable by delta>
CLAMP variable to [0, 100]
```

### Example Rules

```
IF user_message contains gratitude keywords → affection += 2
IF user_message contains negative emotion keywords → mood = "concerned", trust += 1
IF user_message contains hostility keywords → affection -= 3, trust -= 2
IF conversation_length > 10 turns → intimacy += 1 per turn
IF user_message length < 5 words for 3 consecutive turns → energy -= 5
```

## State as Prompt Context

The current state is injected into the system prompt as:

```
Current relationship state:
affection=62, trust=70, mood=concerned, intimacy=55, energy=65
```

The model uses this to calibrate its response style.

## State Persistence

- State resets at the start of each new session (for MVP).
- Future work: persist state across sessions using a state file or database.
