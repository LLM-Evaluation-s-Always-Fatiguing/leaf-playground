import json
from typing import List, Set

from pydantic import BaseModel


def _get_value(data: dict, field_chain: List[str]):
    field = field_chain.pop(0)
    value = data[field]
    if field_chain:
        return _get_value(value, field_chain)
    return value


class Data(BaseModel):
    """
    Super class for all data structures that can be observed by agents.
    """

    def format(self, template: str, fields: Set[str]) -> str:
        """
        Using a template to formate values of all the selected fields into a beautiful and readable string.

        :param template: a template string used to formate values
        :type template: str
        :param fields: the selected fields whose values are required by the given template
        :type fields: List[str]
        :return: a formatted string
        :rtype: str
        """
        raw_data = json.loads(self.model_dump_json())
        data = dict()
        for field in fields:
            field_chain = field.split(".")
            field_key = "_".join(field_chain)
            data[field_key] = _get_value(raw_data, field_chain)

        return template.format(**data)
