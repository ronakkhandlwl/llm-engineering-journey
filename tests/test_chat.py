import builtins

import pytest

from llm_client.base import ROLE_ASSISTANT, ROLE_USER, ChatResult, Message
from llm_client.chat import format_usage, run_chat


class FakeProvider:
    """In-memory provider that records the history it was sent."""

    name = "fake"
    model = "gpt-4o-mini"

    def __init__(self):
        self.calls: list[list[Message]] = []

    def chat(self, messages, system=None):
        # Snapshot the history passed for this turn.
        self.calls.append(list(messages))
        return ChatResult(
            text=f"reply-{len(self.calls)}",
            model=self.model,
            input_tokens=10,
            output_tokens=5,
        )


def _scripted_input(lines):
    it = iter(lines)

    def fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration as exc:  # mimic Ctrl-D
            raise EOFError from exc

    return fake_input


@pytest.mark.unit
def test_format_usage_includes_tokens_and_cost():
    line = format_usage("gpt-4o-mini", 1_000_000, 0, 0.15)
    assert "in=1000000" in line
    assert "out=0" in line
    assert "est_cost=$0.150000" in line
    assert "session_total=$0.150000" in line


@pytest.mark.unit
def test_multi_turn_history_accumulates(monkeypatch, capsys):
    provider = FakeProvider()
    monkeypatch.setattr(builtins, "input", _scripted_input(["hi", "again", "exit"]))

    run_chat(provider)

    # Two model calls (third line is the exit word).
    assert len(provider.calls) == 2
    # First call sees one user turn.
    assert provider.calls[0] == [Message(role=ROLE_USER, content="hi")]
    # Second call sees the full history: user, assistant, user.
    assert provider.calls[1] == [
        Message(role=ROLE_USER, content="hi"),
        Message(role=ROLE_ASSISTANT, content="reply-1"),
        Message(role=ROLE_USER, content="again"),
    ]

    out = capsys.readouterr().out
    assert "reply-1" in out
    assert "est_cost=" in out


@pytest.mark.unit
def test_failed_call_drops_unanswered_turn(monkeypatch, capsys):
    class BoomProvider(FakeProvider):
        def chat(self, messages, system=None):
            raise RuntimeError("boom")

    provider = BoomProvider()
    # First turn errors, then exit; a healthy provider would have history again.
    monkeypatch.setattr(builtins, "input", _scripted_input(["hi", "exit"]))

    run_chat(provider)

    out = capsys.readouterr().out
    assert "[error]" in out
    assert "boom" in out
