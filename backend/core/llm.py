"""
llm.py — LLM Service Wrapper

This module provides a streamlined interface for interacting with the Gemini API.
"""

from google import genai
from . import config

class LLMService:
    """
    Service for interacting with the Gemini API.
    """
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in .env or environment.")
        
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_MODEL

    def generate_content(self, prompt: str, temperature: float = None) -> str:
        """
        Generate content using the Gemini API.
        """
        temp = temperature if temperature is not None else config.GEMINI_TEMPERATURE
        
        print(f"[llm] Calling Gemini ({self.model})...")
        completion = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={'temperature': temp}
        )
        return completion.text.strip()
