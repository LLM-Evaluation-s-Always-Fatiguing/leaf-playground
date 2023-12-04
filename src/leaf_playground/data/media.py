import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Union, Literal

from pydantic import Field, model_validator

from .base import Data


class MediaType(Enum):
    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class Media(Data, ABC):
    display_text: str = Field(default=...)
    type: str = Field(default="unknown")

    @model_validator(mode='before')
    def set_type(cls, values):
        if values.get('type') is not None:
            values['type'] = cls.__name__.lower()
            print(f'set type: {values["type"]} for {values["display_text"]}')
        if values.get('display_text') is None:
            values['display_text'] = cls._generate_display_text(values)
        return values

    @classmethod
    def _generate_display_text(cls, values) -> str:
        pass

    def __str__(self):
        return self.display_text

    class Config:
        frozen = True
        extra = "forbid"


class Text(Media):
    text: str = Field(default=...)

    @classmethod
    def _generate_display_text(cls, values) -> str:
        return values['text'][:256]

    def __repr__(self):
        return f"<Text {self.text}>"


class Json(Media):
    data: Union[dict, list] = Field(default={})

    @classmethod
    def _generate_display_text(cls, values) -> str:
        if isinstance(values['data'], dict) and values['data'].get('text') is not None:
            return values['data']['text'][:256]
        return 'Json Object' if isinstance(values['data'], dict) else 'Json Array'

    def __repr__(self):
        return f"<Json {json.dumps(self.data, ensure_ascii=False, indent=2)}>"


class Image(Media):
    pass


class Audio(Media):
    pass


class Video(Media):
    pass


__all__ = [
    "Media",
    "Text",
    "Json",
    "Image",
    "Audio",
    "Video",
    "MediaType",
]
