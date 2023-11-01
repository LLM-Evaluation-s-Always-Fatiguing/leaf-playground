import importlib
import importlib.machinery
from functools import partial
from types import ModuleType
from typing import Optional, Any

from pydantic import BaseModel, Field, FilePath


class DynamicObject(BaseModel):
    obj: str = Field(default=...)
    module: Optional[str] = Field(default=None)
    source_file: Optional[FilePath] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.module is None and self.source_file is None:
            raise ValueError("at least one of 'module' or 'source_file' should not None.")
        if self.module and self.source_file:
            raise ValueError("can't specify 'module' and 'source_file' at the same time.")


def dynamically_import_obj(o: DynamicObject):
    if o.module is not None:
        module = importlib.import_module(o.module)
    else:
        module = importlib.machinery.SourceFileLoader(
            o.source_file.name, o.source_file.as_posix()
        ).load_module()
    return module.__dict__[o.obj]


class DynamicFn(BaseModel):
    fn: DynamicObject = Field(default=...)
    default_kwargs: Optional[dict] = Field(default=None)


def dynamically_import_fn(f: DynamicFn):
    fn = dynamically_import_obj(f.fn)
    if f.default_kwargs is not None:
        fn = partial(fn, **f.default_kwargs)
    return fn
