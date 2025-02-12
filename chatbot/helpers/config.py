import os
import json
from dotenv import load_dotenv
from typing import Dict, Any

class Configuration:
    """Manages configuration and settings for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        # self.api_key = os.getenv("OPENAI_API_KEY")
        # self.api_key = os.getenv("GITHUB_API_KEY")

        self.model_owner = os.getenv("LLM_SOURCE", "openai")

        if (self.model_owner == "openai"):
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = "gpt-4o"
            self.api_key = os.getenv("OPENAI_API_KEY")
        else:
            if os.getenv("IN_CONTAINER", "false") == "true":
                self.endpoint = "http://host.docker.internal:11434/v1/chat/completions"
            else:
                self.endpoint = "http://localhost:11434/v1/chat/completions"
            self.model = "llama3.2"
            self.api_key = ""

        if os.getenv("SERVER_CONFIG_FILE", None):
            with open(os.getenv("SERVER_CONFIG_FILE"), 'r') as f:
                self.mcp_servers_config = json.load(f)
        else:
            self.mcp_servers_config = {
                "mcpServers": {}
            }

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()
