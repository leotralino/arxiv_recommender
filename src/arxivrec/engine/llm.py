import os
from abc import abstractmethod
from typing import Optional

import ollama


class BaseLLM:
    def __init__(self, model_name: str, options: dict, **kwargs):
        self.model_name = model_name
        self.options = options

    @abstractmethod
    def call(self, prompt):
        """
        Message is just a huge string.
        """
        pass


class OLlamaLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
    ):
        self.model_name = model_name

    def call(self, prompt):
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            format="json",
            options={"temperature": 0},
        )

        return response


class OpenaiLLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-5.1", api_key: str = None):
        super().__init__(model_name)
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
