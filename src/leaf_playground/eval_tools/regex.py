import re
from typing import List

from pydantic import Field

from .base import EvalTool, EvalToolConfig


class RegexEvalToolConfig(EvalToolConfig):
    regex_rules: List[str] = Field(
        default=[r"\b([A-D])\b"], description="Regex rules for extracting answers from the scene."
    )
    ignore_case: bool = Field(
        default=True, description="Whether to ignore case when matching regex rules and answers."
    )


class RegexEvalTool(EvalTool):
    config_obj = RegexEvalToolConfig
    config: config_obj

    def __init__(self, config: config_obj):
        super().__init__(config)
        self._regex_rules = config.regex_rules
        self._ignore_case = config.ignore_case

    def extract_answer(self, origin_answer: str) -> str:
        regex_flag = re.IGNORECASE if self.ignore_case else 0
        for rule in self._regex_rules:
            match = re.search(rule, origin_answer, flags=regex_flag)
            if match:
                return match.group(1)
        return ""

    @property
    def ignore_case(self) -> bool:
        return self._ignore_case
