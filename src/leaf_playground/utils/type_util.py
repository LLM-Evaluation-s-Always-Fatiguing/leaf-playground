import asyncio
from functools import wraps, partial
from typing import Any, Callable, Type

from pydantic import BaseModel, ConfigDict, ValidationError


def validate_type(var: Any, expect_type: Type) -> bool:
    class Model(BaseModel):
        model_config = ConfigDict(strict=True, arbitrary_types_allowed=True)

        value: expect_type

    try:
        Model(value=var)
        return True
    except ValidationError:
        return False


class MethodOutputsTypeChecker:
    def _type_check_wrap(self, func: Callable = None, *, outputs_type: Type = None):
        if func is None:
            return partial(self._type_check_wrap, outputs_type=outputs_type)

        def validate(outputs: Any):
            if outputs_type is None and outputs is not None:
                raise TypeError(f"expected outputs of function {func.__name__} is None, got {outputs}")
            elif not validate_type(outputs, outputs_type):
                raise TypeError(
                    f"expected outputs type/annotation of function {func.__name__} is {outputs_type}, "
                    f"got an invalid outputs {outputs} with type {type(outputs)}"
                )
            return outputs

        @wraps(func)
        def wrap(*args, **kwargs):
            result = func(*args, **kwargs)
            return validate(result)

        @wraps(func)
        async def awrap(*args, **kwargs):
            result = await func(*args, **kwargs)
            return validate(result)

        return awrap if asyncio.iscoroutinefunction(func) else wrap

    def __init__(self, func: Callable = None, *, outputs_type: Type = None):
        self._wrap = self._type_check_wrap(func=func, outputs_type=outputs_type)
        self._func = func

    def __call__(self, *args, **kwargs):
        if not self._func:
            self._func = args[0]
            self._wrap = self._wrap(func=self._func)
            return self
        return self._wrap(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


__all__ = ["validate_type", "MethodOutputsTypeChecker"]
