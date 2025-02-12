import logging
import requests

from typing import Dict, List
from helpers.config import Configuration


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, config: Configuration) -> None:
        self.config = config

    def get_response(self, messages: List[Dict[str, str]], tools) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: A list of message dictionaries.
            
        Returns:
            The LLM's response as a string.
            
        Raises:
            RequestException: If the request to the LLM fails.
        """
        url = self.config.endpoint

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        payload = {
            "messages": messages,
            "model": self.config.model,
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None
        }

        if len(tools) >= 1:
            payload["tools"] = tools
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']
            
        except requests.exceptions.RequestException as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            
            if e.response is not None:
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")
                
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."
