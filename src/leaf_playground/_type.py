from typing import Generic, TypeVar


T = TypeVar("T")


class Immutable(Generic[T]):
    def __init__(self, value):
        self._value = value

        def __setattr__(cls, __name, __value):
            raise AttributeError("You are changing an immutable variable")

        self.__setattr__ == __setattr__

    def __get__(self, instance, owner):
        return self._value

    def __set__(self, instance, value):
        raise AttributeError("You are changing an immutable variable")

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            return getattr(self._value, item)

    def __getattribute__(self, item):
        try:
            return super().__getattribute__(item)
        except:
            return getattr(self._value, item)

    def __getitem__(self, item):
        return self._value.__getitem__(item)

    def __setitem__(self, key, value):
        raise AttributeError("You are changing an immutable variable")

    def __iter__(self):
        return self._value

    def __next__(self):
        return self._value.__next__()

    def __bool__(self):
        return bool(self._value)

    def __hash__(self):
        return self.__hash__()


__all__ = ["Immutable"]
