import json
from abc import ABC
from typing import Literal, Optional, Union, Any

from pydantic import Field
from typing_extensions import Annotated

from .base import Data


class Media(Data, ABC):
    display_text: Optional[str] = Field(default=None)
    type: Literal["media"] = Field(default="media")

    def model_post_init(self, __context: Any) -> None:
        if not self.display_text:
            self.display_text = self._generate_display_text()

    def _generate_display_text(self) -> str:
        return None

    def __str__(self):
        return self.display_text


class Text(Media):
    text: str = Field(default=..., frozen=True)
    type: Literal["text"] = Field(default="text")

    def _generate_display_text(self) -> str:
        return self.text[:256]

    def __repr__(self):
        return f"<Text {self.text}>"


class Json(Media):
    data: Union[dict, list] = Field(default={}, frozen=True)
    type: Literal["json"] = Field(default="json")

    def _generate_display_text(self) -> str:
        if isinstance(self.data, dict) and self.data.get('text') is not None:
            return self.data['text'][:256]
        return 'Json Object' if isinstance(self.data, dict) else 'Json Array'

    def __repr__(self):
        return f"<Json {json.dumps(self.data, ensure_ascii=False, indent=2)}>"


class Image(Media):
    type: Literal["image"] = Field(default="image")


class Audio(Media):
    type: Literal["audio"] = Field(default="audio")


class Video(Media):
    type: Literal["video"] = Field(default="video")


MediaType = Annotated[
    Union[Text, Json, Audio, Image, Video],
    Field(discriminator="type")
]


__all__ = [
    "Media",
    "Text",
    "Json",
    "Image",
    "Audio",
    "Video",
    "MediaType",
]
