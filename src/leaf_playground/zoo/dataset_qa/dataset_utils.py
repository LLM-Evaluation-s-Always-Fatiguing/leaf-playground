from typing import Optional, List

from datasets import load_dataset
from pydantic import Field

from ..._config import _Config
from ...utils.import_util import dynamically_import_fn, DynamicFn


class DatasetConfig(_Config):
    path: str = Field(default=...)
    split: str = Field(default=...)
    question_column: str = Field(default=...)
    golden_answer_column: Optional[str] = Field(default=None)
    filter_conditions: Optional[List[DynamicFn]] = Field(default=None)
    question_preprocessor: Optional[DynamicFn] = Field(default=None)
    name: Optional[str] = Field(default=None)
    data_dir: Optional[str] = Field(default=None)
    data_files: Optional[List[str]] = Field(default=None)


def prepare_dataset(config: DatasetConfig) -> List[dict]:
    dataset = load_dataset(
        path=config.path,
        split=config.split,
        name=config.name,
        data_dir=config.data_dir,
        data_files=config.data_files,
        keep_in_memory=True
    )
    for condition in (config.filter_conditions or []):
        dataset = dataset.filter(
            function=dynamically_import_fn(condition),
            batched=True,
            keep_in_memory=True
        )
    if config.question_preprocessor:
        dataset = dataset.map(
            function=dynamically_import_fn(config.question_preprocessor),
            batched=True,
            keep_in_memory=True
        )
    return dataset.to_list()
