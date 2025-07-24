"""Run this model in Python

> pip install mistralai

This script provides an interactive chat session with a Mistral model
hosted on GitHub's inference service. It maintains conversation history
for context-aware interactions.
"""
import os
import sys
import json
from mistralai import Mistral, UserMessage, AssistantMessage

# --- Configuration ---
SERVER_URL = "https://models.github.ai/inference"
DEFAULT_MODEL = "mistral-ai/Codestral-2501"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.1
HISTORY_FILE = "chat_history.json"
# ---------------------

def get_GH_TOKEN():
    """
    Retrieves the GitHub token from the environment variables.
    Exits the script with an error message if the token is not found.
    """
    token = os.environ.get("GH_TOKEN")
    if not token:
        print("Error: The 'GH_TOKEN' environment variable is not set.", file=sys.stderr)
        print("Please create a Personal Access Token (PAT) on GitHub and set it.", file=sys.stderr)
        sys.exit(1)
    return token

def load_history():
    """Loads chat history from file."""
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
            # Convert the loaded history into Message objects
            messages = [
                UserMessage(content=msg['content']) if msg['role'] == 'user' else AssistantMessage(content=msg['content'])
                for msg in history
            ]
        return messages
    except FileNotFoundError:
        return []

def save_history(messages):
    """Saves chat history to file."""
    history = [{'role': 'user', 'content': msg.content} if isinstance(msg, UserMessage) \
               else {'role': 'assistant', 'content': msg.content} for msg in messages]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def main():
    """
    Main function to run the interactive chat client.
    
    To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
    Create your PAT token by following instructions here: 
    https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    """
    api_key = get_GH_TOKEN()

    try:
        client = Mistral(
            server_url=SERVER_URL,
            api_key=api_key,
        )
    except Exception as e:
        print(f"Error creating Mistral client: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting interactive chat with model: {DEFAULT_MODEL}")
    print("Type 'quit' or 'exit' to end the session (or press Ctrl+C).")
    print("-" * 20)

    # This list will store the conversation history
    messages = load_history()
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting chat.")
                break
            
            # Skip empty inputs
            if not user_input:
                continue

            messages.append(UserMessage(content=user_input))

            response = client.chat.complete(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
            )

            if response.choices and len(response.choices) > 0:
                assistant_response = response.choices[0].message
                print(f"Assistant: {assistant_response.content}")
                messages.append(assistant_response)  # Add assistant's reply to history
                save_history(messages)  # Save after each turn
            else:
                print("Error: No response received from the model.", file=sys.stderr)
                # Remove the last user message since we couldn't get a response
                if messages:
                    messages.pop()

        except Exception as e:
            # Check if it's a Mistral-related error
            if "mistral" in str(e).lower() or "api" in str(e).lower():
                print(f"\nMistral API error: {e}", file=sys.stderr)
            else:
                print(f"\nUnexpected error: {e}", file=sys.stderr)
            # Remove the last user message to allow retrying
            if messages:
                messages.pop()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    main()
