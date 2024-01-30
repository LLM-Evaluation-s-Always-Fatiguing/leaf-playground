import warnings
from enum import Enum

from pydantic import BaseModel, Field


class IDEType(Enum):
    PyCharm = "pycharm"
    VSCode = "vscode"


class DebuggerConfig(BaseModel):
    ide_type: IDEType = Field(default=IDEType.PyCharm)
    host: str = Field(default="localhost")
    port: int = Field(default=3456)
    debug: bool = Field(default=False)


def try_to_import_pydevd_pycharm():
    try:
        import pydevd_pycharm
    except ImportError:
        warnings.warn("try to import pydevd-pycharm but failed, make sure you have it installed.")
        return None
    else:
        return pydevd_pycharm


def maybe_set_debugger(config: DebuggerConfig, patch_multiprocessing: bool = False):
    if not config.debug:
        return
    if config.ide_type == IDEType.VSCode:
        raise NotImplementedError()
    else:
        pydevd_pycharm = try_to_import_pydevd_pycharm()
        if not pydevd_pycharm:
            return
        pydevd_pycharm.settrace(
            config.host,
            port=config.port,
            stdoutToServer=True,
            stderrToServer=True,
            patch_multiprocessing=patch_multiprocessing
        )
