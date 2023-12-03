import random
from typing import Optional, List, Any

from datasets import load_dataset
from pydantic import Field

from leaf_playground._config import _Config
from leaf_playground.utils.import_util import dynamically_import_fn, DynamicFn


class DatasetConfig(_Config):
    path: str = Field(default="explodinggradients/fiqa")
    split: str = Field(default="baseline")
    question_column: str = Field(default="question")
    golden_answer_column: str = Field(default="answer")
    ground_truth_column: str = Field(default="ground_truths")
    num_questions: int = Field(default=-1)
    filter_conditions: Optional[List[DynamicFn]] = Field(default=None)
    question_preprocessor: Optional[DynamicFn] = Field(default=None)
    name: Optional[str] = Field(default=None)
    data_dir: Optional[str] = Field(default=None)
    data_files: Optional[List[str]] = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.num_questions < -1 or self.num_questions == 0:
            raise ValueError(f"num_questions should be -1 or positive, got {self.num_questions}")


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
    dataset = dataset.to_list()
    if config.num_questions != -1:
        data_indices = list(range(len(dataset)))
        dataset = [dataset[i] for i in random.sample(data_indices, min(len(data_indices), config.num_questions))]
    return dataset


__all__ = [
    "prepare_dataset",
    "DatasetConfig"
]
