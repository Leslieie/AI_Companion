"""Tests for Phase 2 architecture modules.

Covers StateTracker, MemoryStore, PolicySelector, PromptBuilder,
and EmotionClassifier without loading any model.
"""

import pytest

from src.modules.state_tracker import StateTracker, CompanionState
from src.modules.memory_store import MemoryStore
from src.modules.policy_selector import select_policy
from src.modules.emotion_classifier import classify_emotion
from src.inference.prompt_builder import build_prompt


# ---------------------------------------------------------------------------
# StateTracker
# ---------------------------------------------------------------------------

class TestStateTracker:
    def test_defaults(self):
        tracker = StateTracker()
        state = tracker.get_state()
        assert state["affection"] == 50
        assert state["trust"] == 50
        assert state["intimacy"] == 50
        assert state["mood"] == "neutral"
        assert state["energy"] == 70

    def test_sad_keyword_sets_mood_concerned(self):
        tracker = StateTracker()
        state = tracker.update("I feel really sad today")
        assert state["mood"] == "concerned"

    def test_negative_emotion_increases_trust(self):
        tracker = StateTracker()
        state = tracker.update("I am so stressed out")
        assert state["trust"] == 51

    def test_gratitude_increases_affection(self):
        tracker = StateTracker()
        state = tracker.update("thank you so much")
        assert state["affection"] == 52

    def test_hostility_decreases_affection_and_trust(self):
        tracker = StateTracker()
        state = tracker.update("just leave me alone already")
        assert state["affection"] == 47
        assert state["trust"] == 48

    def test_positive_emotion_sets_cheerful(self):
        tracker = StateTracker()
        state = tracker.update("I am so happy right now")
        assert state["mood"] == "cheerful"
        assert state["affection"] == 51

    def test_lively_increases_energy(self):
        tracker = StateTracker()
        state = tracker.update("that was so funny haha")
        assert state["energy"] == 72

    def test_intimacy_growth_after_ten_turns(self):
        tracker = StateTracker()
        for _ in range(10):
            tracker.update("just chatting along with you today")
        assert tracker.get_state()["intimacy"] == 50
        state = tracker.update("one more turn of chatting today")
        assert state["intimacy"] == 51

    def test_energy_decay_on_short_messages(self):
        tracker = StateTracker()
        tracker.update("ok")
        tracker.update("sure")
        state = tracker.update("yep")
        assert state["energy"] == 65

    def test_energy_decay_resets_on_long_message(self):
        tracker = StateTracker()
        tracker.update("ok")
        tracker.update("sure")
        tracker.update("actually I had a really wonderful and productive day")
        assert tracker.get_state()["energy"] == 70

    def test_clamp_upper_bound(self):
        tracker = StateTracker()
        tracker.state.affection = 99
        state = tracker.update("thank you so very much friend")
        assert state["affection"] == 100

    def test_clamp_lower_bound(self):
        tracker = StateTracker()
        tracker.state.affection = 1
        state = tracker.update("you are so annoying leave me alone")
        assert state["affection"] == 0

    def test_reset(self):
        tracker = StateTracker()
        tracker.update("I feel really sad and stressed")
        tracker.update("thank you for being kind to me")
        tracker.reset()
        state = tracker.get_state()
        assert state == {
            "affection": 50,
            "trust": 50,
            "intimacy": 50,
            "mood": "neutral",
            "energy": 70,
        }
        assert tracker.turn_count == 0

    def test_turn_count_increments(self):
        tracker = StateTracker()
        tracker.update("hello there friend")
        tracker.update("how are you doing today")
        assert tracker.turn_count == 2


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class TestMemoryStore:
    def test_add_and_retrieve(self):
        store = MemoryStore()
        store.add("User is a student", 1)
        assert store.retrieve("anything") == ["User is a student"]

    def test_retrieve_returns_most_recent(self):
        store = MemoryStore()
        store.add("fact A", 1)
        store.add("fact B", 2)
        store.add("fact C", 3)
        store.add("fact D", 4)
        result = store.retrieve("anything", top_k=2)
        assert result == ["fact C", "fact D"]

    def test_no_duplicates(self):
        store = MemoryStore()
        store.add("User likes hiking", 1)
        store.add("User likes hiking", 2)
        assert store.size() == 1

    def test_empty_string_ignored(self):
        store = MemoryStore()
        store.add("", 1)
        assert store.size() == 0

    def test_clear(self):
        store = MemoryStore()
        store.add("fact A", 1)
        store.add("fact B", 2)
        store.clear()
        assert store.size() == 0
        assert store.retrieve("anything") == []

    def test_capacity_eviction(self):
        store = MemoryStore()
        for i in range(55):
            store.add(f"fact {i}", i)
        assert store.size() == 50
        assert store.retrieve("anything", top_k=1) == ["fact 54"]
        facts = [m["fact"] for m in store.long_term_memories]
        assert "fact 0" not in facts

    def test_memory_dict_structure(self):
        store = MemoryStore()
        store.add("User studies at CMU", 5)
        entry = store.long_term_memories[0]
        assert entry["fact"] == "User studies at CMU"
        assert entry["source_turn"] == 5
        assert "timestamp" in entry

    def test_extract_and_store_with_trigger(self):
        store = MemoryStore()
        result = store.extract_and_store("I am a grad student", 1)
        assert result == "I am a grad student"
        assert store.size() == 1

    def test_extract_and_store_i_like(self):
        store = MemoryStore()
        result = store.extract_and_store("I like hiking on weekends", 2)
        assert result == "I like hiking on weekends"

    def test_extract_and_store_no_trigger(self):
        store = MemoryStore()
        result = store.extract_and_store("hello there", 1)
        assert result is None
        assert store.size() == 0

    def test_extract_and_store_case_insensitive(self):
        store = MemoryStore()
        result = store.extract_and_store("I AM a CMU student", 1)
        assert result is not None

    def test_retrieve_empty_store(self):
        store = MemoryStore()
        assert store.retrieve("anything") == []


# ---------------------------------------------------------------------------
# PolicySelector
# ---------------------------------------------------------------------------

class TestPolicySelector:
    def _default_state(self, **overrides) -> dict:
        state = {
            "affection": 50, "trust": 50, "intimacy": 50,
            "mood": "neutral", "energy": 70,
        }
        state.update(overrides)
        return state

    def test_sad_returns_comforting(self):
        assert select_policy("sad", self._default_state()) == "comforting"

    def test_anxious_returns_comforting(self):
        assert select_policy("anxious", self._default_state()) == "comforting"

    def test_lonely_returns_comforting(self):
        assert select_policy("lonely", self._default_state()) == "comforting"

    def test_happy_returns_playful(self):
        assert select_policy("happy", self._default_state()) == "playful"

    def test_angry_returns_tense(self):
        assert select_policy("angry", self._default_state()) == "tense"

    def test_neutral_returns_neutral(self):
        assert select_policy("neutral", self._default_state()) == "neutral"

    def test_concerned_mood_returns_reflective(self):
        assert select_policy("neutral", self._default_state(mood="concerned")) == "reflective"

    def test_low_energy_returns_reflective(self):
        assert select_policy("neutral", self._default_state(energy=20)) == "reflective"

    def test_emotion_takes_priority_over_state(self):
        assert select_policy("happy", self._default_state(mood="concerned")) == "playful"


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def _default_state(self) -> dict:
        return {
            "affection": 60, "trust": 70, "intimacy": 55,
            "mood": "concerned", "energy": 65,
        }

    def test_returns_list_of_dicts(self):
        history = [{"role": "user", "content": "hello"}]
        result = build_prompt(history, self._default_state(), [], "neutral")
        assert isinstance(result, list)
        assert all(isinstance(m, dict) for m in result)

    def test_system_message_is_first(self):
        history = [{"role": "user", "content": "hello"}]
        result = build_prompt(history, self._default_state(), [], "neutral")
        assert result[0]["role"] == "system"

    def test_history_appended_after_system(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you"},
        ]
        result = build_prompt(history, self._default_state(), [], "neutral")
        assert len(result) == 4
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "hello"
        assert result[2]["role"] == "assistant"
        assert result[3]["content"] == "how are you"

    def test_system_prompt_contains_persona(self):
        history = [{"role": "user", "content": "hello"}]
        result = build_prompt(history, self._default_state(), [], "neutral")
        system = result[0]["content"]
        assert "Ari" in system

    def test_system_prompt_contains_state(self):
        history = [{"role": "user", "content": "hello"}]
        result = build_prompt(history, self._default_state(), [], "neutral")
        system = result[0]["content"]
        assert "affection=60" in system
        assert "trust=70" in system
        assert "intimacy=55" in system
        assert "mood=concerned" in system
        assert "energy=65" in system

    def test_system_prompt_contains_memories(self):
        history = [{"role": "user", "content": "hello"}]
        memories = ["User is a student", "User likes hiking"]
        result = build_prompt(history, self._default_state(), memories, "neutral")
        system = result[0]["content"]
        assert "User is a student" in system
        assert "User likes hiking" in system

    def test_system_prompt_no_memories_shows_none(self):
        history = [{"role": "user", "content": "hello"}]
        result = build_prompt(history, self._default_state(), [], "neutral")
        system = result[0]["content"]
        assert "- none" in system

    def test_system_prompt_contains_policy(self):
        history = [{"role": "user", "content": "hello"}]
        result = build_prompt(history, self._default_state(), [], "comforting")
        system = result[0]["content"]
        assert "comforting" in system


# ---------------------------------------------------------------------------
# EmotionClassifier
# ---------------------------------------------------------------------------

class TestEmotionClassifier:
    def test_happy(self):
        assert classify_emotion("I am so happy today") == "happy"

    def test_sad(self):
        assert classify_emotion("I feel really sad") == "sad"

    def test_anxious(self):
        assert classify_emotion("I am so stressed about exams") == "anxious"

    def test_angry(self):
        assert classify_emotion("I am so frustrated with this") == "angry"

    def test_lonely(self):
        assert classify_emotion("I feel so alone tonight") == "lonely"

    def test_neutral(self):
        assert classify_emotion("What should we talk about") == "neutral"

    def test_case_insensitive(self):
        assert classify_emotion("I am EXCITED about this") == "happy"

    def test_empty_string(self):
        assert classify_emotion("") == "neutral"
