import logging
from typing import Dict, Any

class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Dict[str, Any] = input_schema

        logging.debug(f"Loaded tool: {self.name} with schema {self.input_schema}")

    def format_for_llm(self) -> str:
        """Format tool information for LLM.
        
        Returns:
            A formatted string describing the tool.
        """

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema
        }
