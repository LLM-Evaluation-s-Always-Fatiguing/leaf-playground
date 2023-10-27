from typing import Set

from pydantic import BaseModel


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
        return template.format(**self.model_dump(include=fields))
