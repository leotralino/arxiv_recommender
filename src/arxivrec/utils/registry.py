import logging

logger = logging.getLogger(__name__)


class Registry:
    """A generic registry to store and retrieve classes."""

    def __init__(self, name: str):
        self.name = name
        self._classes = {}

        logger.info(f"Registry {self.name} created!")

    def register(self, key: str):
        """Decorator to register a class under a specific key."""

        def decorator(cls):
            self._classes[key] = cls
            logger.info(f"Registry {self.name} has a new children {cls}!")
            return cls

        return decorator

    def get(self, key: str):
        if key not in self._classes:
            raise ValueError(
                f"'{key}' not found in {self.name} registry."
                f"Available: {list(self._classes.keys())}"
            )
        return self._classes[key]

    def __getitem__(self, key: str):
        return self.get(key)
