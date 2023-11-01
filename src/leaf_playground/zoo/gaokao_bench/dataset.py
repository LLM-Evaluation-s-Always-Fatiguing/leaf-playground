from typing import List, Any

from datasets import load_dataset, concatenate_datasets, Dataset
from pydantic import BaseModel, Field

DATASET_PATH = "AsakusaRinne/gaokao_bench"
DS_TYPE2TEMPLATE = {
    "2010-2013_English_MCQs": {
        "teacher_prompt": "[Single Choice] Reading the given sentence or conversation, "
                          "choose a word or phrase from the given candidates to make it "
                          "complete and syntax correct.\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "The answer is:",
        "student_fields": None
    },
    "2010-2022_Biology_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Chemistry_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Chinese_Lang_and_Usage_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Chinese_Modern_Lit": {
        "teacher_prompt": "{question}",
        "teacher_fields": {"question"},
        "student_prompt": "每个问题对应答案的选项序号分别是：",
        "student_fields": None
    },
    "2010-2022_English_Fill_in_Blanks": {
        "teacher_prompt": "{question}",
        "teacher_fields": {"question"},
        "student_prompt": "各空白处对应答案的选项序号依次是：",
        "student_fields": None
    },
    "2010-2022_English_Reading_Comp": {
        "teacher_prompt": "Below is an article and some questions. Reading the article and "
                          "answer the questions based on the information you learn from the "
                          "article by choosing the correct option.\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "The answer to each questions are:",
        "student_fields": None
    },
    "2010-2022_Geography_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_History_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Math_I_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Math_II_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Physics_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2010-2022_Political_Science_MCQs": {
        "teacher_prompt": "【单选】阅读题目，从给定的选项中选择正确的一项。\n{question}",
        "teacher_fields": {"question"},
        "student_prompt": "答案是：",
        "student_fields": None
    },
    "2012-2022_English_Cloze_Test": {
        "teacher_prompt": "{question}",
        "teacher_fields": {"question"},
        "student_prompt": "各空白处对应填入的选项按顺序依次是：",
        "student_fields": None
    }
}
DATASET_SPLIT = "dev"
DATASET_FIELDS = [
    "question",
    "year",
    "answer",
    "type"
]
YEARS = [str(year) for year in range(2010, 2023)]
DS_TYPES = list(DS_TYPE2TEMPLATE.keys())


class DatasetConfig(BaseModel):
    types: List[str] = Field(default=DS_TYPES)
    years: List[str] = Field(default=YEARS)

    def model_post_init(self, __context: Any) -> None:
        if not all(year in YEARS for year in self.years):
            raise ValueError(f"years must be a subset of {YEARS}, but get {self.years}")
        if not all(type_ in DS_TYPES for type_ in self.types):
            raise ValueError(f"types must be a subset of {DS_TYPES}, but get {self.types}")


def load_hf_dataset(config: DatasetConfig) -> Dataset:
    dataset = None
    for ds_type in DS_TYPE2TEMPLATE:
        ds: Dataset = load_dataset(
            path=DATASET_PATH,
            name=ds_type,
            split=DATASET_SPLIT,
            keep_in_memory=True
        )
        if ds_type not in config.types:
            continue
        ds = ds.add_column(name="type", column=[ds_type] * ds.num_rows)
        if dataset is None:
            dataset = ds
        else:
            dataset = concatenate_datasets(dsets=[dataset, ds])

    dataset = dataset.filter(
        lambda example: str(example["year"]) in config.years,
        keep_in_memory=True
    )
    dataset = dataset.select_columns(column_names=DATASET_FIELDS)

    return dataset


def format_data(data: dict) -> List[str]:
    template = DS_TYPE2TEMPLATE[data["type"]]
    teacher_msg = template["teacher_prompt"].format(**{f: data[f] for f in template["teacher_fields"]})
    student_msg = template["student_prompt"]

    return [teacher_msg, student_msg]
