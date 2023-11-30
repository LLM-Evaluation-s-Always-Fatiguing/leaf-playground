from .base import *


def simple_agg_sum(values):
    return sum(values)


def simple_agg_mean(values):
    return sum(values) / len(values)


def simple_agg_max(values):
    return max(values)


def simple_agg_min(values):
    return min(values)


def simple_agg_median(values):
    return sorted(values)[len(values) // 2]
