from typing import Dict, List, Union


def simple_nested_sum(values: List[Union[float, Dict[str, float]]]) -> Union[float, Dict[str, float]]:
    nested = isinstance(values[0], dict)
    if nested:
        return {key: simple_nested_sum([item[key] for item in values]) for key in values[0].keys()}
    return sum(values)


def simple_nested_mean(values: List[Union[float, Dict[str, float]]]) -> Union[float, Dict[str, float]]:
    nested = isinstance(values[0], dict)
    if nested:
        return {key: simple_nested_mean([item[key] for item in values]) / len(values) for key in values[0].keys()}
    return sum(values) / len(values)


def simple_nested_max(values: List[Union[float, Dict[str, float]]]) -> Union[float, Dict[str, float]]:
    nested = isinstance(values[0], dict)
    if nested:
        return {key: simple_nested_max([item[key] for item in values]) for key in values[0].keys()}
    return max(values)


def simple_nested_min(values: List[Union[float, Dict[str, float]]]) -> Union[float, Dict[str, float]]:
    nested = isinstance(values[0], dict)
    if nested:
        return {key: simple_nested_min([item[key] for item in values]) for key in values[0].keys()}
    return min(values)


def simple_nested_median(values: List[Union[float, Dict[str, float]]]) -> Union[float, Dict[str, float]]:
    nested = isinstance(values[0], dict)
    if nested:
        return {key: simple_nested_median([item[key] for item in values]) for key in values[0].keys()}
    return sorted(values)[len(values) // 2]


__all__ = [
    "simple_nested_sum",
    "simple_nested_mean",
    "simple_nested_max",
    "simple_nested_min",
    "simple_nested_median",
]
