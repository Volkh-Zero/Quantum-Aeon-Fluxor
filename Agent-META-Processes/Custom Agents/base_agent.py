import json
import os

import re

class BaseAgent:
    def __init__(self, agent_name):
        # Sanitize agent_name to remove invalid filename characters
        safe_agent_name = re.sub(r'[<>:"/\\|?*]', '_', agent_name)
        self.agent_name = safe_agent_name # This line was missing a closing parenthesis for os.path.join
        self.chat_history_path = os.path.join("chat_histories", f"{self.agent_name}_history.json")
    def load_chat_history(self):
        if os.path.exists(self.chat_history_path):
            try:
                with open(self.chat_history_path, "r") as f:
                    self.chat_history = json.load(f)
            except (IOError, PermissionError) as e:
                print(f"Error reading chat history file: {e}")
                self.chat_history = []
            except json.JSONDecodeError as e:
                print(f"Error decoding chat history JSON: {e}")
                self.chat_history = []
        else:
            self.chat_history = []
    def save_chat_history(self):
        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(self.chat_history_path), exist_ok=True)
        try:
            with open(self.chat_history_path, "w") as f:
                json.dump(self.chat_history, f, indent=2)
        except (OSError, IOError) as e:
            print(f"Error saving chat history: {e}")
                self.chat_history = json.load(f)
        else:
            self.chat_history = []

    def save_chat_history(self):
        with open(self.chat_history_path, "w") as f:
            json.dump(self.chat_history, f, indent=2)

    def add_to_history(self, user_message, bot_response):
        self.chat_history.append({"user": user_message, "bot": bot_response})
        self.save_chat_history()
