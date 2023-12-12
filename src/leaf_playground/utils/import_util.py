import importlib
import importlib.machinery
import inspect
import glob
import os
from functools import partial
from hashlib import md5
from pathlib import Path
from typing import Any, List, Optional, Type

from pydantic import BaseModel, Field, FilePath


_IMPORTED_OBJECTS = {}
_IMPORTED_FUNCTIONS = {}


class DynamicObject(BaseModel):
    obj: str = Field(default=...)
    module: Optional[str] = Field(default=None)
    source_file: Optional[FilePath] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.module is None and self.source_file is None:
            raise ValueError("at least one of 'module' or 'source_file' should not None.")
        if self.module and self.source_file:
            raise ValueError("can't specify 'module' and 'source_file' at the same time.")

    @property
    def hash(self) -> str:
        obj = self.obj
        module = "" if not self.module else self.module
        source_file = "" if not self.source_file else self.source_file.as_posix()
        return md5((obj + module + source_file).encode(encoding="utf-8")).hexdigest()

    @classmethod
    def create_dynamic_obj(cls, obj: Type) -> "DynamicObject":
        dynamic_obj = cls(obj=obj.__name__, source_file=Path(inspect.getfile(obj)))
        _IMPORTED_OBJECTS[dynamic_obj.hash] = obj
        return dynamic_obj

    @classmethod
    def bind_dynamic_obj(cls, obj: "DynamicObject", target: Type) -> None:
        _IMPORTED_OBJECTS[obj.hash] = target


def dynamically_import_obj(o: DynamicObject):
    if o.hash in _IMPORTED_OBJECTS:
        return _IMPORTED_OBJECTS[o.hash]
    if o.module is not None:
        module = importlib.import_module(o.module)
    else:
        module = importlib.machinery.SourceFileLoader(
            o.source_file.name, o.source_file.as_posix()
        ).load_module()
    obj = module.__dict__[o.obj]
    _IMPORTED_OBJECTS[o.hash] = obj
    return obj


class DynamicFn(BaseModel):
    fn: DynamicObject = Field(default=...)
    default_kwargs: Optional[dict] = Field(default=None)

    @property
    def hash(self) -> str:
        return self.fn.hash + md5(str(self.default_kwargs).encode(encoding="utf-8")).hexdigest()


def dynamically_import_fn(f: DynamicFn):
    if f.hash in _IMPORTED_FUNCTIONS:
        return _IMPORTED_FUNCTIONS[f.hash]
    fn = dynamically_import_obj(f.fn)
    if f.default_kwargs is not None:
        fn = partial(fn, **f.default_kwargs)
    _IMPORTED_FUNCTIONS[f.hash] = fn
    return fn


def find_subclasses(package_path: str, base_class: Type) -> List[DynamicObject]:
    classes = []

    for file in glob.glob(os.path.join(package_path, f'**/*.py'), recursive=True):
        module_path = Path(file)
        module = importlib.machinery.SourceFileLoader(
            module_path.name, module_path.as_posix()
        ).load_module()

        if not hasattr(module, "__all__"):
            continue
        # TODO: figure out why using module.__dict__ directly will get duplicate objects
        for name in module.__all__:
            obj = module.__dict__[name]
            if not inspect.isclass(obj):
                continue
            if issubclass(obj, base_class) and obj != base_class and obj.__module__ == module.__name__:
                classes.append(DynamicObject(obj=obj.__name__, source_file=module_path))

    return classes


__all__ = [
    "DynamicObject",
    "DynamicFn",
    "dynamically_import_obj",
    "dynamically_import_fn",
    "find_subclasses",
]
