from enum import Enum

from pydantic import Field

from .base import Data


class Media(Data):
    pass


class Text(Media):
    text: str = Field(default=...)

    def __str__(self):
        return self.text


class Json(Media):
    pass


class Image(Media):
    pass


class Audio(Media):
    pass


class Video(Media):
    pass


class MediaType(Enum):
    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

    @classmethod
    def get_media_obj(cls, media_type: "MediaType"):
        if not isinstance(media_type, cls):
            raise TypeError(f"media_type should be an instance of {cls.__name__}")
        if media_type.value == "text":
            return Text
        elif media_type.value == "json":
            return Json
        elif media_type.value == "image":
            return Image
        elif media_type.value == "audio":
            return Audio
        elif media_type.value == "video":
            return Video
        else:
            raise ValueError(f"unknown media type: {media_type}")


__all__ = [
    "Media",
    "Text",
    "Json",
    "Image",
    "Audio",
    "Video",
    "MediaType",
]
