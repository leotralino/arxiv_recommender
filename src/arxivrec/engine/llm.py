import os
from abc import abstractmethod

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
    def __init__(
        self,
        model_name: str = "openai/gpt-5.1-instant",
        api_key: str = "",
        api_base: str = "",
        options: dict | None = None,
        **kwargs,
    ):
        super().__init__(model_name, options)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.kwargs = kwargs

    def call(self, prompt: str) -> dict:
        from litellm import completion

        messages = []
        messages.append({"role": "user", "content": prompt})

        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self.options:
            completion_kwargs.update(self.options)
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base
        if self.kwargs:
            completion_kwargs.update(self.kwargs)

        response = completion(**completion_kwargs)
        content = response.choices[0].message.content
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))

        return {"response": content, "raw_response": response}
