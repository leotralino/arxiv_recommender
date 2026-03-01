from types import SimpleNamespace

from arxivrec.engine.llm import LLM_REGISTRY


def test_ollama_call_uses_registered_class_and_default_options(monkeypatch):
    captured = {}

    def fake_generate(*, model, prompt, format, options):
        captured.update(
            {
                "model": model,
                "prompt": prompt,
                "format": format,
                "options": options,
            }
        )
        return {"response": '{"papers": []}'}

    monkeypatch.setattr("arxivrec.engine.llm.ollama.generate", fake_generate)

    client = LLM_REGISTRY["ollama"](model_name="llama3.2:3b")
    response = client.call("rank these papers")

    assert response["response"] == '{"papers": []}'
    assert captured == {
        "model": "llama3.2:3b",
        "prompt": "rank these papers",
        "format": "json",
        "options": {"temperature": 0},
    }


def test_openai_call_uses_registered_class_and_forwards_kwargs(monkeypatch):
    call_kwargs = {}

    class FakeResponse:
        choices = [SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]

    def fake_completion(**kwargs):
        call_kwargs.update(kwargs)
        return FakeResponse()

    fake_litellm = SimpleNamespace(completion=fake_completion)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_litellm)

    client = LLM_REGISTRY["openai"](
        model_name="openai/gpt-5.1-instant",
        api_key="test-key",
        api_base="http://localhost:8000/v1",
        options={"temperature": 0.3},
        timeout=30,
    )
    response = client.call("hello")

    assert response["response"] == '{"ok": true}'
    assert response["raw_response"].__class__.__name__ == "FakeResponse"
    assert call_kwargs["model"] == "openai/gpt-5.1-instant"
    assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["api_base"] == "http://localhost:8000/v1"
    assert call_kwargs["timeout"] == 30
