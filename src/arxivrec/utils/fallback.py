import functools
import logging

logger = logging.getLogger(__name__)


def fallback(arg_name: str, fallback_values: list):
    """
    Generic decorator to retry a method with different values for a specific kwarg.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # 1. Try the original call first
            result = func(self, *args, **kwargs)

            # Check if result is "successful" (not empty/None)
            # Adjust the 'if' condition based on what your methods return
            if result is not None and not (hasattr(result, "empty") and result.empty):
                return result

            # 2. Iterate through fallbacks
            for value in fallback_values:
                logger.warning(f"Retrying {func.__name__} with {arg_name}={value}")

                # Update the specific keyword argument
                kwargs[arg_name] = value

                result = func(self, *args, **kwargs)

                if result is not None and not (
                    hasattr(result, "empty") and result.empty
                ):
                    return result

            return result

        return wrapper

    return decorator
