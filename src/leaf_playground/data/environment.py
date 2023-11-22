import random
from abc import abstractmethod
from typing import Any, List, Set, Optional

from pydantic import Field

from .base import Data


class EnvironmentVariable(Data):
    """
    Environment variable of a scene that can be updated in different ways

    :param name: the environment variable's name
    :type name: str
    :param description: the environment variable's description
    :type description: str
    :param current_value: the value that the environment variable holds for current phase
    :type current_value: Any
    """
    name: str = Field(default=...)
    description: str = Field(default=...)
    current_value: Any = Field(default=...)

    @abstractmethod
    def update(self, *args, **kwargs):
        return


class RandomnessEnvironmentVariable(EnvironmentVariable):
    """
    A subtype of EnvironmentVariable whose current_value is updated by choose value from a space randomly

    :param random_space: a value space that current_value will be chosen fom
    :type random_space: Set[Any]
    :param current_value: the value that the environment variable holds for current phase
    :type current_value: Optional[Any]
    """
    random_space: Set[Any] = Field(default=...)
    current_value: Optional[Any] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.current_value is None:
            self.update()
        if self.current_value not in self.random_space:
            raise ValueError(
                f"value of 'current_value' must in 'random_space'({self.random_space}), "
                f"but get {self.current_value}."
            )

    def update(self, exclude_current: bool = False) -> None:
        """
        Update current_value

        :param exclude_current: whether exclude current value from random space when choose a new value
        :return: None
        """
        choices = self.random_space
        if exclude_current:
            choices -= {self.current_value}
        self.current_value = random.choice(list(choices))


class ChainedEnvironmentVariable(EnvironmentVariable):
    """
    A subtype of EnvironmentVariable whose potential values is defined in a chain and
    will be updated in the order of candidate values in the chain.

    :param chain: a chain that contains potential values and defines their order
    :type chain: List[Any]
    :param current_value: the value that the environment variable holds for current phase
    :type current_value: Optional[Any]
    :param recurrent: whether the chain is a recurrent chain
    :type recurrent: bool
    """
    chain: List[Any] = Field(default=...)
    current_value: Optional[Any] = Field(default=None)
    recurrent: bool = Field(default=False)

    def model_post_init(self, __context: Any) -> None:
        if self.current_value is None:
            self.current_value = self.chain[0]
        if self.current_value not in self.chain:
            raise ValueError(
                f"value of 'current_value' must in 'chain'({self.chain}), "
                f"but get {self.current_value}."
            )

    def update(self) -> None:
        current_position = self.chain.index(self.current_value)
        next_position = (current_position + 1) % len(self.chain)
        if current_position < next_position or self.recurrent:
            self.current_value = self.chain[next_position]


class NumericEnvironmentVariable(EnvironmentVariable):
    """
    A subtype of EnvironmentVariable whose value can be updated by using numeric operators.

    :param current_value: the value that the environment variable holds for current phase
    :type current_value: float
    :param le: the environment variable's maximum value, if is None, there is no maximum constrain
    :type le: Optional[float]
    :param ge: the environment variable's minimum value, if is None, there is no minimum constrain
    """

    current_value: float = Field(default=...)
    le: Optional[float] = Field(default=None)
    ge: Optional[float] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._check_value_legality(value=self.current_value, value_name="current_value")

    def _check_value_legality(self, value: float, value_name: str = "value") -> None:
        legal = True
        if self.le is not None and value > self.le:
            legal = False
        if self.ge is not None and value < self.ge:
            legal = False
        if not legal:
            raise ValueError(
                f"'{value_name}'({self.current_value}) isn't legal for "
                f"'le'={self.le} and 'ge'={self.ge}"
            )

    def update(self, new_value: float) -> None:
        self._check_value_legality(value=new_value, value_name="new_value")
        self.current_value = new_value

    def __add__(self, other: float):
        new_value = self.current_value + other
        self.update(new_value)

    def __sub__(self, other: float):
        new_value = self.current_value - other
        self.update(new_value)


class ConstantEnvironmentVariable(EnvironmentVariable):
    """
    A subtype of EnvironmentVariable whose values are unchangeable.
    """

    def update(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} is unchangeable.")

    def __setattr__(self, key, value):
        if key in self.__dict__:
            raise TypeError(f"{self.__class__.__name__}.{key} is unchangeable.")


__all__ = [
    "EnvironmentVariable",
    "RandomnessEnvironmentVariable",
    "ChainedEnvironmentVariable",
    "NumericEnvironmentVariable",
    "ConstantEnvironmentVariable"
]
