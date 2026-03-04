import time
from functools import wraps


def _verbosity_level(instance) -> int:
    """Interpret legacy boolean verbosity as levels, defaulting to 0."""
    if instance is None or not hasattr(instance, "verbose"):
        return 0
    value = getattr(instance, "verbose")
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1 if value else 0

def timeit(func):
    """
    A decorator to time member functions, printing the function's name,
    execution time in seconds, minutes, and hours if the class has a 'verbose'
    attribute set to True.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed_time = end_time - start_time
        elapsed_minutes = elapsed_time / 60
        elapsed_hours = elapsed_minutes / 60

        instance = args[0] if args else None
        if _verbosity_level(instance) >= 3:
            class_name = instance.__class__.__name__ if instance else "UnknownClass"
            print(f"Class '{class_name}', Function '{func.__name__}' executed in {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes, {elapsed_hours:.2f} hours).")

        return result

    return wrapper
