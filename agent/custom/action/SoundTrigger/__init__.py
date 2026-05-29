__all__ = ["Ear", "Dodger"]


def __getattr__(name: str):
    if name == "Ear":
        from .SoundListener import Ear
        return Ear
    if name == "Dodger":
        from .DodgeCounterTrigger import Dodger
        return Dodger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
