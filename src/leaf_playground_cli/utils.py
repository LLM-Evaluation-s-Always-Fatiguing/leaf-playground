from os.path import dirname, exists, isfile, join
from sys import maxsize
from typing import Optional


class DirNotFoundError(OSError):
    pass


def get_project_root_path(current: str, max_layers: Optional[int] = None):
    if max_layers is None:
        max_layers = maxsize
    if isfile(current):
        current = dirname(current)
    if exists(join(current,  ".leaf")):
        return current
    parent = dirname(current)
    max_layers -= 1
    if max_layers < 0 or not parent or parent == current:
        raise DirNotFoundError(
            f"Can't find .leaf dir in any ancestor dirs, maybe this is not a leaf project, "
            f"or considering increase max_layers"
        )
    return get_project_root_path(parent, max_layers)


def get_static_dir(current: str, max_layers: Optional[int] = None):
    root_dir = get_project_root_path(current, max_layers)
    static_dir = join(root_dir, "static")
    if not exists(static_dir):
        raise DirNotFoundError(f"static dir not found in project {root_dir}")
    return static_dir


__all__ = ["get_project_root_path", "get_static_dir"]
