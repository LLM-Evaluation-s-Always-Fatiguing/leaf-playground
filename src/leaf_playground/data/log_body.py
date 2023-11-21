from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from .base import Data
from .media import Media, MediaType
from .message import MessageType


class LogBody(Data):
    index: int = Field(default=...)
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    references: Optional[List[MessageType]] = Field(default=None)
    response: MessageType = Field(default=...)
    media_type: MediaType = Field(default=...)
    ground_truth: Optional[Media] = Field(default=None)
    eval_result: Optional[Dict[str, Union[bool, float, str]]] = Field(default=None)
    narrator: Optional[str] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        media_obj = MediaType.get_media_obj(self.media_type)
        if self.references:
            for msg in self.references:
                if not isinstance(msg.content, media_obj):
                    raise ValueError(
                        f"media type of reference message's content should be {media_obj.__name__}, "
                        f"but got {msg.content.__class__.__name__}"
                    )
        if not isinstance(self.response.content, media_obj):
            raise ValueError(
                f"media type of response message's content should be {media_obj.__name__}, "
                f"but got {self.response.content.__class__.__name__}"
            )
        if self.ground_truth and not isinstance(self.ground_truth, media_obj):
            raise ValueError(
                f"media type of ground truth should be {media_obj.__name__}, "
                f"but got {self.ground_truth.__class__.__name__}"
            )


__all__ = ["LogBody"]
