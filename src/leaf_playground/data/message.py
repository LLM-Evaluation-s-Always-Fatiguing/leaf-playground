from datetime import datetime
from typing import Dict, List, Literal, Union
from uuid import uuid4, UUID

from pydantic import Field
from typing_extensions import Annotated

from .base import Data
from .media import Media, Text, Json, Image, Audio, Video
from .profile import Profile
from .._type import Singleton


class Message(Data):
    id_: UUID = Field(default_factory=uuid4)
    sender: Profile = Field(default=...)
    content: Media = Field(default=...)
    receivers: List[Profile] = Field(default=...)
    msg_type: Literal["basic"] = Field(default="basic")
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())

    @property
    def id(self):
        return self.sender_id + "_" + self.id_.hex

    @property
    def sender_name(self):
        return self.sender.name

    @property
    def sender_id(self):
        return self.sender.id

    @property
    def sender_role(self):
        return self.sender.role.name

    @property
    def receiver_names(self):
        return [receiver.name for receiver in self.receivers]

    @property
    def receiver_ids(self):
        return [receiver.id for receiver in self.receivers]

    @property
    def receiver_roles(self):
        return list(set([receiver.role.name for receiver in self.receivers]))


class TextMessage(Message):
    content: Text = Field(default=...)
    msg_type: Literal["text"] = Field(default="text")


class JsonMessage(Message):
    content: Json = Field(default=...)
    msg_type: Literal["json"] = Field(default="json")


class ImageMessage(Message):
    content: Image = Field(default=...)
    msg_type: Literal["image"] = Field(default="image")


class AudioMessage(Message):
    content: Audio = Field(default=...)
    msg_type: Literal["audio"] = Field(default="audio")


class VideoMessage(Message):
    content: Video = Field(default=...)
    msg_type: Literal["video"] = Field(default="video")


class MessagePool(Singleton):
    """
    A global message pool to cache all agents' messages in one engine run

    :param messages: messages that all agents send in one engine run
    :type messages: Dict[str, Message]
    """

    # TODO: fix: design message pool as a singleton and allow get_instance globally may let agents to cheat
    def __init__(self):
        self.messages: Dict[str, Message] = {}

    def clear(self):
        """Clear cached messages, at the start of each engine run, this method should be called"""
        self.messages = {}

    def put_message(self, message: Message):
        """Put one message into the cache"""
        self.messages[message.id] = message

    def get_messages(self, agent: Profile) -> List[Message]:
        """
        Get messages that sent by the agent or is visible by the agent

        :param agent: agent who want to retrieve messages
        :type agent: Profile
        :return: a list of messages that are sent by or visible by the agent
        :rtype: List[Message]
        """
        messages = []
        for message in self.messages.values():
            receivers_ids = [receiver.id for receiver in message.receivers]
            if agent.id in receivers_ids:
                messages.append(message)
        return messages

    def get_message_by_id(self, msg_id: str) -> Message:
        return self.messages[msg_id]


BasicMessageType = Annotated[
    Union[TextMessage, JsonMessage, ImageMessage, AudioMessage, VideoMessage], Field(discriminator="msg_type")
]

__all__ = [
    "Message",
    "TextMessage",
    "JsonMessage",
    "ImageMessage",
    "AudioMessage",
    "VideoMessage",
    "MessagePool",
    "BasicMessageType",
]
