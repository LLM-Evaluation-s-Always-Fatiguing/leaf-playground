from abc import ABC, abstractmethod
from pandas import DataFrame


class BaseVegaChart(ABC):

    @abstractmethod
    def generate(self, data: DataFrame) -> dict:
        pass
