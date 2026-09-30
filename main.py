"""Terminal front end for the chess assistant.

The agent itself moved to `agent.py`, because the Discord bot needs the same
loop and two copies of an agent loop are two sets of bugs. The names are
re-exported here so existing imports -- including the tests -- keep working.

    python main.py                 # talk to it in the terminal
    python discord_bot.py          # same agent, in Discord
"""

import sys

from google import genai

from agent import (  # noqa: F401 - re-exported for the tests and for callers
    FUNCTION_MAP,
    MAX_TOOL_ROUNDS,
    MODEL,
    MODEL_FALLBACK_HINT,
    TOOLS,
    AgentSession,
    create_chat,
    get_api_key,
    run_agent,
)


def main() -> None:
    """Simple interactive loop: prompt, print answer, quit on exit/quit."""
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    try:
        chat = create_chat(client)
    except Exception as e:  # noqa: BLE001 - CLI boundary: report and exit, never traceback
        print(f"Could not start Gemini chat with model '{MODEL}': {e}")
        print(f"Tip: try GEMINI_MODEL={MODEL_FALLBACK_HINT} in your .env")
        sys.exit(1)

    print("Chess agent ready! Ask e.g. 'show me the last 5 games of hikaru'.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bye!")
            break
        try:
            # The terminal keeps the old raw chat (not an AgentSession): one
            # long-lived conversation is exactly what a terminal session is, and
            # a mid-conversation reset would be baffling with no UI to explain it.
            answer, _tools, artifacts = run_agent(chat, user_input)
        except Exception as e:  # noqa: BLE001 - CLI boundary: keep the session alive
            # Friendly error, keep the chat history for the next question.
            print(f"Agent error: {e}")
            print("Tip: if the model says 404/not found, it may be retired.")
            print(f"Try GEMINI_MODEL={MODEL_FALLBACK_HINT} in your .env\n")
            continue
        print(f"\nAgent:\n{answer}\n")
        for tool, path in artifacts:
            # In a terminal a path is the only way to get at the picture, and the
            # model has been told to print them itself -- so this is a
            # belt-and-braces list rather than the only one.
            print(f"  [{tool}] {path}")
        if artifacts:
            print()


if __name__ == "__main__":
    main()
