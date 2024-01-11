class Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonClass(metaclass=Singleton):
    @classmethod
    def get_instance(cls):
        try:
            return cls._instances[cls]
        except KeyError:
            raise KeyError(
                f"class [{cls.__name__}] not instantiate yet, please create one instance before calling this method."
            )


__all__ = ["SingletonClass"]
