from typing import Generic, TypeVar

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass


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


class SingletonMetaClass(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(metaclass=SingletonMetaClass):
    @classmethod
    def get_instance(cls):
        try:
            return cls._instances[cls]
        except KeyError:
            raise KeyError(
                f"class [{cls.__name__}] not instantiate yet, please create one instance before calling this method."
            )


class SingletonBaseModelMetaclass(ModelMetaclass):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonBaseModel(BaseModel, metaclass=SingletonBaseModelMetaclass):
    @classmethod
    def get_instance(cls):
        try:
            return cls._instances[cls]
        except KeyError:
            raise KeyError(
                f"class [{cls.__name__}] not instantiate yet, please create one instance before calling this method."
            )


__all__ = ["Immutable", "Singleton", "SingletonBaseModel"]
