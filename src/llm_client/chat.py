"""Multi-turn chat loop with a swappable backend.

Run:
    python -m llm_client.chat --provider anthropic
    python -m llm_client.chat --provider openai
    python -m llm_client.chat --provider gemini

Conversation history is kept in a plain list of Messages and resent on every
turn, so the model sees the full context. Token usage and an estimated cost are
printed after each reply, with a running session total.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from llm_client.base import ROLE_ASSISTANT, ROLE_USER, LLMProvider, Message
from llm_client.factory import PROVIDERS, get_provider
from llm_client.pricing import estimate_cost

DEFAULT_SYSTEM = "You are a helpful assistant. Be concise."
EXIT_WORDS = {"exit", "quit"}


def format_usage(
    model: str, input_tokens: int, output_tokens: int, session_total: float
) -> str:
    """One-line usage/cost log for a single call."""
    cost = estimate_cost(model, input_tokens, output_tokens)
    return (
        f"[usage] in={input_tokens} out={output_tokens} "
        f"est_cost=${cost:.6f} | session_total=${session_total:.6f}"
    )


def run_chat(provider: LLMProvider, system: str = DEFAULT_SYSTEM) -> None:
    """Interactive REPL loop against a single provider."""
    history: list[Message] = []
    session_total = 0.0
    print(
        f"Chatting with {provider.name} ({provider.model}). "
        f"Type 'exit'/'quit' or Ctrl-D to leave."
    )

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            return

        if user_input.lower() in EXIT_WORDS:
            print("bye 👋")
            return
        if not user_input:
            continue

        history.append(Message(role=ROLE_USER, content=user_input))
        try:
            result = provider.chat(history, system=system)
        except Exception as exc:  # noqa: BLE001 — surface any backend error to the user
            # Drop the unanswered turn so history stays consistent.
            history.pop()
            print(f"[error] {provider.name} call failed: {exc}")
            continue

        history.append(Message(role=ROLE_ASSISTANT, content=result.text))
        session_total += estimate_cost(
            result.model, result.input_tokens, result.output_tokens
        )

        print(f"\n{provider.name}> {result.text}")
        print(
            format_usage(
                result.model, result.input_tokens, result.output_tokens, session_total
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn chat across LLM providers."
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS),
        default="anthropic",
        help="Which backend to use (default: anthropic).",
    )
    args = parser.parse_args()

    load_dotenv()
    provider = get_provider(args.provider)
    run_chat(provider)


if __name__ == "__main__":
    main()
