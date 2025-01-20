"""
Reads and stores application configuration (tokens, settings, etc.).
"""

import os
from configparser import ConfigParser


class ConfigManager:
    """
    Manages the application's configuration.
    Reads settings from config.ini and provides
    them via properties/methods.
    """

    def __init__(self, config_path: str = "config.ini"):
        self._config = ConfigParser()
        self._config.read(config_path)

        self.TELEGRAM_TOKEN = self._config.get("tokens", "TELEGRAM_TOKEN")
        self.OPENAI_API_KEY = self._config.get("tokens", "OPENAI_API_KEY")
        self.TAVILY_API_KEY = self._config.get("tokens", "TAVILY_API_KEY")

        self.RANDOM_RESPONSE_PROBABILITY = self._config.getfloat(
            "settings", "RANDOM_RESPONSE_PROBABILITY"
        )
        self.LLM_DECISSION_TO_RESPOND_THRESHOLD = self._config.getfloat(
            "settings", "LLM_DECISSION_TO_RESPOND_THRESHOLD"
        )
        
        self.MESSAGE_HISTORY_LIMIT = self._config.getint("settings", "MESSAGE_HISTORY_LIMIT")

        # Set environment variables if needed
        os.environ["TAVILY_API_KEY"] = self.TAVILY_API_KEY

        self.MAIN_WORKFLOW_MODEL = self._config.get("models", "MAIN_WORKFLOW_MODEL")
        self.SHOULD_RESPOND_MODEL = self._config.get("models", "SHOULD_RESPOND_MODEL")
        self.VOICE_TRANSCTIPTION_MODEL = self._config.get("models", "VOICE_TRANSCTIPTION_MODEL")


# Create a single shared instance for the entire application
CONFIG = ConfigManager()
