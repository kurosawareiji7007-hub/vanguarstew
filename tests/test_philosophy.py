"""Tests for the maintainer-philosophy step (issue #11 few-shot examples). Run:

    VANGUARSTEW_OFFLINE=1 python -m pytest -q
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ["VANGUARSTEW_OFFLINE"] = "1"

from agent.llm import LLM  # noqa: E402
from agent.philosophy import (  # noqa: E402
    _OFFLINE_STUB,
    FEWSHOT,
    _normalize_philosophy,
    _normalize_string_list,
    _normalize_text,
    infer_philosophy,
    render_philosophy_for_prompt,
)

EXPECTED_KEYS = {"summary", "values", "merge_bar", "direction", "evidence"}


def _fewshot_outputs():
    """The JSON object on the line after each 'OUTPUT:' marker (single-line examples)."""
    outs = []
    for chunk in FEWSHOT.split("OUTPUT:\n")[1:]:
        outs.append(json.loads(chunk.splitlines()[0]))
    return outs


def test_fewshot_examples_present_and_valid():
    outputs = _fewshot_outputs()
    assert len(outputs) >= 1  # acceptance: prompt includes 1-2 examples
    for ex in outputs:
        assert EXPECTED_KEYS <= set(ex), f"missing keys: {EXPECTED_KEYS - set(ex)}"
        assert isinstance(ex["values"], list) and ex["values"]
        assert isinstance(ex["evidence"], list) and ex["evidence"]
        assert isinstance(ex["summary"], str) and ex["summary"]


def test_infer_philosophy_offline_has_expected_keys():
    llm = LLM(api_key="offline")
    out = infer_philosophy({"recent_commits": [{"subject": "init"}]}, llm)
    assert EXPECTED_KEYS <= set(out)


class _ListLLM:
    """A model that answers with a top-level JSON array instead of an object."""

    def chat_json(self, system, user, stub=None):
        return ["conservative", "stability-over-features"]


def test_infer_philosophy_coerces_non_dict_response_to_stub():
    out = infer_philosophy({"recent_commits": [{"subject": "init"}]}, _ListLLM())
    assert isinstance(out, dict)
    assert EXPECTED_KEYS <= set(out)


def test_normalize_text_coerces_scalars():
    assert _normalize_text(None, "fallback") == "fallback"
    assert _normalize_text("ship fixes", "fallback") == "ship fixes"
    assert _normalize_text(42, "fallback") == "42"


def test_normalize_string_list_coerces_to_string_list():
    assert _normalize_string_list(None) == []
    assert _normalize_string_list("conservative") == ["conservative"]
    assert _normalize_string_list(["a", "", None, 7]) == ["a", "7"]
    assert _normalize_string_list({"bad": True}) == []


def test_normalize_philosophy_maps_malformed_fields():
    stub = {
        "summary": "offline stub philosophy",
        "values": [],
        "merge_bar": "unknown (offline)",
        "direction": "unknown (offline)",
        "evidence": [],
    }
    out = _normalize_philosophy({
        "summary": None,
        "values": "feature-first",
        "merge_bar": 123,
        "direction": None,
        "evidence": None,
    }, stub)
    assert out["summary"] == "offline stub philosophy"
    assert out["values"] == ["feature-first"]
    assert out["merge_bar"] == "123"
    assert out["direction"] == "unknown (offline)"
    assert out["evidence"] == []


class _MalformedPhilosophyLLM:
    offline = False

    def chat_json(self, system, user, stub=None):
        return {
            "summary": None,
            "values": "conservative",
            "merge_bar": "high bar",
            "direction": 99,
            "evidence": "recent refactors",
        }


def test_infer_philosophy_normalizes_malformed_field_types():
    out = infer_philosophy({"recent_commits": [{"subject": "init"}]}, _MalformedPhilosophyLLM())
    assert isinstance(out["summary"], str)
    assert isinstance(out["values"], list)
    assert isinstance(out["evidence"], list)
    assert out["values"] == ["conservative"]
    assert out["direction"] == "99"
    assert out["evidence"] == ["recent refactors"]

def test_infer_philosophy_handles_non_dict_context():
    llm = LLM(api_key="offline")
    for bad_context in (None, "not a dict", 42, [], True):
        out = infer_philosophy(bad_context, llm)
        assert EXPECTED_KEYS <= set(out), f"missing keys for context={bad_context!r}: {EXPECTED_KEYS - set(out)}"
        assert out["values"] == [], f"values should be [] not {out['values']!r}"
        assert out["merge_bar"] == _OFFLINE_STUB["merge_bar"]
        assert out["direction"] == _OFFLINE_STUB["direction"]
        assert out["evidence"] == []


def test_infer_philosophy_non_dict_context_returns_fresh_copy():
    llm = LLM(api_key="offline")
    a = infer_philosophy(None, llm)
    b = infer_philosophy(None, llm)
    a["summary"] = "mutated"
    assert b["summary"] != "mutated"


def _over_cap_philosophy():
    return {
        "summary": "A mature library that guards stability and a small dependency surface.",
        "values": ["conservative", "stability-over-features", "evidence-based"],
        "merge_bar": "Merges fixes and well-justified changes; rejects new deps.",
        "direction": "Incremental hardening on the 3.x line.",
        "evidence": [f"evidence item number {i}: " + ("x" * 80) for i in range(40)],
    }


def test_render_philosophy_for_prompt_is_byte_identical_when_under_cap():
    philosophy = {
        "summary": "short",
        "values": ["a"],
        "merge_bar": "bar",
        "direction": "dir",
        "evidence": ["one"],
    }
    expected = json.dumps(philosophy, indent=1)
    assert len(expected) < 4000
    assert render_philosophy_for_prompt(philosophy, 4000, indent=1) == expected
    # Compact dump (review call site) stays byte-identical too.
    compact = json.dumps(philosophy)
    assert render_philosophy_for_prompt(philosophy, 1500) == compact


def test_render_philosophy_for_prompt_over_cap_keeps_valid_json_and_core_fields():
    """#1962: hard-slicing mid-evidence produced unterminated JSON in scored prompts."""
    philosophy = _over_cap_philosophy()
    full = json.dumps(philosophy, indent=1)
    assert len(full) > 4000

    for cap in (4000, 3000, 1500):
        # The old hard-slice path is invalid JSON.
        try:
            json.loads(full[:cap])
            raise AssertionError(f"expected hard-slice at {cap} to be invalid JSON")
        except json.JSONDecodeError:
            pass

        rendered = render_philosophy_for_prompt(philosophy, cap, indent=1)
        assert len(rendered) <= cap
        parsed = json.loads(rendered)  # must be valid JSON
        assert parsed["summary"] == philosophy["summary"]
        assert parsed["values"] == philosophy["values"]
        assert parsed["merge_bar"] == philosophy["merge_bar"]
        assert parsed["direction"] == philosophy["direction"]
        assert isinstance(parsed["evidence"], list)
        assert len(parsed["evidence"]) < len(philosophy["evidence"])

    # Input philosophy is not mutated — solve()'s returned object stays intact.
    assert len(philosophy["evidence"]) == 40


def test_render_philosophy_for_prompt_review_compact_over_cap():
    philosophy = _over_cap_philosophy()
    rendered = render_philosophy_for_prompt(philosophy, 1500)
    assert len(rendered) <= 1500
    parsed = json.loads(rendered)
    assert parsed["summary"] == philosophy["summary"]
    assert isinstance(parsed["evidence"], list)
