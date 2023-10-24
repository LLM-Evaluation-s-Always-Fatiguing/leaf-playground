from abc import ABC
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Data:
    """
    Super class for all data structures that can be observed by agents.
    """

    @staticmethod
    def _filter_fields(
        data: dict,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> dict:
        if include is None and exclude is None:
            return data

        fields = list(data.keys())
        if include is not None:
            fields = [f for f in fields if f in include]
        if exclude is not None:
            fields = [f for f in fields if f not in exclude]
        data = {f: data[f] for f in fields}
        return data

    def to_dict(self, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> dict:
        """
        Convert data structure to a dict

        :param include: fields that should be included in the final dict
        :type include: Optional[List[str]]
        :param exclude: fields that should be excluded in the final dict
        :type exclude: Optional[List[str]]
        :return: a dict contains the data structure's all selected fields' name and value
        :rtype: dict
        """
        data = self.__dict__
        return self._filter_fields(data, include, exclude)

    @classmethod
    def from_dict(cls, data: dict) -> "Data":
        """
        Initialize data structure from a dict

        :param data: a dict used to initialize a data structure
        :type data: dict
        :return: "Observable" type data structure instance
        :rtype: Data
        """
        return cls(**data)

    def format(self, template: str, fields: List[str]) -> str:
        """
        Using a template to formate values of all the selected fields into a beautiful and readable string.

        :param template: a template string used to formate values
        :type template: str
        :param fields: the selected fields whose values are required by the given template
        :type fields: List[str]
        :return: a formatted string
        :rtype: str
        """
        return template.format(**self.to_dict(include=fields))
