import time
from typing import Callable, TypeVar, Any

T = TypeVar('T')

def with_retry(
    fn: Callable[..., T],
    retries: int = 1,
    delay: float = 0.5,
    on_retry: Callable[[Exception], None] = None
) -> T:
    """
    Standard retry decorator/wrapper for agent execution stages.
    """
    last_error = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if i < retries:
                if on_retry:
                    on_retry(e)
                time.sleep(delay)
    raise last_error
