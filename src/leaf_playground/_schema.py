import json
from abc import abstractmethod

from pydantic import BaseModel


class _Schema(BaseModel):

    @abstractmethod
    def valid(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(by_alias=True), f, indent=2)

    @classmethod
    def load(cls, file_path: str):
        kwargs = json.load(open(file_path, "r", encoding="utf-8"))
        return cls(**kwargs)
