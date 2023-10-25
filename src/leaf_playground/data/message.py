import json
from dataclasses import dataclass
from typing import Any, List, Optional
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
    :type timestamp: float
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
    :type timestamp: float
    """
    content: str


@dataclass
class JSONMessage(TextMessage):

    def parse_content(self, required_fields: Optional[List[str]] = None, **kwargs) -> dict:
        """
        Parse message content(must be a json format string) into a dict and check if all required fields are obtained.
        :param required_fields: fields that must be contained in the parsed dict, defaults to None
        :type required_fields: Optional[List[str]]
        :param kwargs: additional key word arguments that will be passed into `json.loads`
        :return: a dictionary parsed from the message content
        :rtype: dict
        """

        data = json.loads(self.content, **kwargs)
        if required_fields and not all(f in data for f in required_fields):
            raise KeyError(
                f"required fields are {required_fields}, but not all of them found in the parsed dict."
            )

        return data


MESSAGE_TYPES = {
    "base": Message,
    "text": TextMessage,
    "json": JSONMessage
}
