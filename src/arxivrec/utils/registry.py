import logging

logger = logging.getLogger(__name__)


class Registry:
    """A generic registry to store and retrieve classes."""

    def __init__(self, name: str):
        self.name = name
        self._classes = {}

        logger.debug(f"Registry {self.name} created!")

    def register(self, key: str):
        """Decorator to register a class under a specific key."""

        def decorator(cls):
            self._classes[key] = cls
            logger.debug(f"Registry {self.name} has a new children {cls}!")
            return cls

        return decorator

    def get(self, key: str):
        if key not in self._classes:
            raise ValueError(
                f"'{key}' not found in {self.name} registry."
                f"Available: {self.show_available()}"
            )
        return self._classes[key]

    def show_available(self) -> list[str]:
        """Returns a list of all registered keys (e.g., ['ollama', 'openai'])."""
        return list(self._classes.keys())

    def __getitem__(self, key: str):
        return self.get(key)

    def __repr__(self):
        return f"Registry({self.name}, available={self.list_available()})"

    def __iter__(self):
        return iter(self._classes)
