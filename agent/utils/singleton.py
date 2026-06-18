class MetaSingleton(type):
    def __call__(cls, *args, **kwargs):
        if "__instance" not in cls.__dict__:
            cls.__instance = super().__call__(*args, **kwargs)
        return cls.__instance


class Singleton(metaclass=MetaSingleton):
    pass
