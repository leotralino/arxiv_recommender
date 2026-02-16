import os
from abc import abstractmethod
from typing import Optional

import ollama

from arxivrec.utils.registry import Registry

LLM_REGISTRY = Registry("LLM")


class BaseLLM:
    def __init__(self, model_name: str, options: dict | None = None, **kwargs):
        self.model_name: str = model_name
        self.options: dict | None = options

    @abstractmethod
    def call(self, prompt):
        """
        Message is just a huge string.
        """
        pass


@LLM_REGISTRY.register("ollama")
class OLlamaLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        options: dict | None = None,
    ):
        super().__init__(model_name, options)

    def call(self, prompt):
        if not self.options:
            self.options = {"temperature": 0}

        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            format="json",
            options=self.options,
        )

        return response


@LLM_REGISTRY.register("openai")
class OpenaiLLM(BaseLLM):
    def __init__(self, model_name: str = "gpt-5.1-instant", api_key: str = ""):
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
