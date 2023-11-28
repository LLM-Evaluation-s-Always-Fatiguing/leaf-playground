import anyio
import asyncio
import sys
from functools import partial
from typing import Any, Callable
if sys.version_info >= (3, 10):  # pragma: no cover
    from typing import ParamSpec
else:  # pragma: no cover
    from typing_extensions import ParamSpec


P = ParamSpec("P")


async def run_asynchronously(func: Callable[P, Any], *args: P.args, **kwargs: P.kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        if kwargs:
            func = partial(func, **kwargs)
        return await anyio.to_thread.run_sync(func, *args)


__all__ = [
    "run_asynchronously"
]
