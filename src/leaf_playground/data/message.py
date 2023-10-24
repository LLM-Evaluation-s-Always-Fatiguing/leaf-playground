from dataclasses import dataclass
from typing import Any
from uuid import UUID

from .base import Data


@dataclass
class Message(Data):
    """
    A data structure that is used among agents to communicate with each other.

    :param sender_id: the id of the agent who send the message
    :type sender_id: UUID
    :param sender_name: the name of the agent who send the message
    :type sender_name: str
    :param content: content of the message
    :type content: Any
    :param timestamp: the timestamp that the message is sent
    :type timestamp: int
    """

    sender_id: UUID
    sender_name: str
    content: Any
    timestamp: float


@dataclass
class TextMessage(Message):
    """
    A subtype of Message whose content is pure text.

    :param sender_id: the id of the agent who send the message
    :type sender_id: UUID
    :param sender_name: the name of the agent who send the message
    :type sender_name: str
    :param content: content of the message
    :type content: str
    :param timestamp: the timestamp that the message is sent
    :type timestamp: int
    """
    content: str


MESSAGE_TYPES = {
    "base": Message,
    "text": TextMessage
}
