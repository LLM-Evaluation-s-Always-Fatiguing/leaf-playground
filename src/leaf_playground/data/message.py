import json
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import Field

import leaf_playground.agent
from .base import Data


class Message(Data):
    """
    A data structure that is used among agents to communicate with each other.

    :param sender_id: the id of the agent who send the message
    :type sender_id: UUID
    :param sender_name: the name of the agent who send the message
    :type sender_name: str
    :param sender_role_name: the name of the agent's role
    :type sender_role_name: str
    :param receiver_ids: the id of agents who can see the message,
        defaults to None, means anyone can see the message
    :type receiver_ids: List[UUID] | None
    :param receiver_names: the name of agents who can see the message,
        defaults to None, means anyone can see the message
    :type receiver_names: List[str] | None
    :param receiver_role_names: the role name of agents who can see the message,
        defaults to None, means anyone can see the message
    :param content: content of the message
    :type content: Any
    :param time: the time that the message is sent
    :type time: datetime
    """

    sender_id: UUID = Field(default=...)
    sender_name: str = Field(default=...)
    sender_role_name: str = Field(default=...)
    receiver_ids: Optional[List[UUID]] = Field(default=None)
    receiver_names: Optional[List[str]] = Field(default=None)
    receiver_role_names: Optional[List[str]] = Field(default=None)
    content: Any = Field(default=...)
    time: datetime = Field(default_factory=lambda: datetime.utcnow())


class MessagePool(Data):
    """
    A global message pool to cache all agents' messages in one engine run

    :param messages: messages that all agents send in one engine run
    :type messages: List[Message]
    """

    messages: List[Message] = Field(default=[])

    def clear(self):
        """Clear cached messages, at the start of each engine run, this method should be called"""
        self.messages = []

    def put_message(self, message: Message):
        """Put one message into the cache"""
        self.messages.append(message)

    def get_messages(self, agent: "leaf_playground.agent.Agent") -> List[Message]:
        """
        Get messages that sent by the agent or is visible by the agent

        :param agent: agent who want to retrieve messages
        :type agent: Agent
        :return: a list of messages that are sent by or visible by the agent
        :rtype: List[Message]
        """
        messages = []
        for message in self.messages:
            if not any([message.receiver_ids, message.receiver_names, message.receiver_role_names]) \
                    or (message.receiver_ids and agent.id in message.receiver_ids) \
                    or (message.receiver_names and agent.name in message.receiver_names) \
                    or (message.receiver_role_names and agent.role_name in message.receiver_role_names) \
                    or agent.id == message.sender_id:
                messages.append(message)
        return messages


class TextMessage(Message):
    """
    A subtype of Message whose content is pure text.

    :param content: content of the message
    :type content: str
    """
    content: str = Field(default=...)


class JSONMessage(TextMessage):

    # TODO: 思考直接继承自基类

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

