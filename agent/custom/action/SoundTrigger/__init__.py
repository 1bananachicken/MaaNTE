__all__ = ["Ear", "Dodger"]


def __getattr__(name: str):
    """Lazy import of module attributes.

    This function allows lazy importing of the Ear and Dodger classes
    when accessed as module attributes.

    Args:
        name (str): The name of the attribute to import.

    Returns:
        The requested class.

    Raises:
        AttributeError: If the attribute name is not in __all__.
    """
    if name == "Ear":
        from .SoundListener import Ear
        return Ear
    if name == "Dodger":
        from .DodgeCounterTrigger import Dodger
        return Dodger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
