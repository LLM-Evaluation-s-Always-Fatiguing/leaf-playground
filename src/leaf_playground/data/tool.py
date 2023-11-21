from typing import Optional

from pydantic import Field

from .base import Data


class Tool(Data):
    pass


class HFTool(Tool):
    repo_id: str = Field(default=...)
    model_repo_id: Optional[str] = Field(default=None)
    token: Optional[str] = Field(default=None)
    remote: bool = Field(default=False)


__all__ = [
    "Tool",
    "HFTool"
]
