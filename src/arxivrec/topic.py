from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    id: str = "AI"
    description: str = "Retrieval-augmented generation (RAG)"
    categories: list[str] = field(default_factory=lambda: ["cs.AI"])
