from enum import Enum

from pydantic import Field

from .dataset_utils import prepare_dataset, DatasetConfig
from ...scene.base import (
    Scene,
    SceneSchema,
    SceneConfig,
    SceneSubClsImplConfig,
    RoleDefinition,
    RoleSchema,
    EnvironmentVariableDefinition,
    EnvironmentSchema
)
from ...utils.import_util import DynamicObject


class RoleName(Enum):
    EXAMINER = "examiner"
    EXAMINEE = "examinee"


DSQASchema = SceneSchema(
    name="dataset_qa",
    description="This scene have one examiner and several examinees, the examiner will "
                "select questions from a given dataset, and the examinees should answer "
                "the questions sent by the examiner.",
    role_schema=RoleSchema(
        num_participants=-1,
        definitions=[
            RoleDefinition(
                name=RoleName.EXAMINER.value,
                description="the one that select a question from a given dataset and send to all examinees",
                role_num=1
            ),
            RoleDefinition(
                name=RoleName.EXAMINEE.value,
                description="the one that answer the question sent by a examiner",
            )
        ]
    ),
    environment_schema=EnvironmentSchema(
        definitions=[]
    )
)


class DSQAConfig(SceneConfig):
    dataset_config: DatasetConfig = Field(defalt=...)


def init(self: Scene):
    self._dataset = prepare_dataset(config=self.config.dataset_config)


def is_terminal(self: Scene):
    return True if not self._dataset else False


def get_data(self: Scene):
    return self._dataset.pop(0)


DSQA = Scene.implement_sub_cls(
    config=SceneSubClsImplConfig(
        cls_name="DatasetQA",
        config_obj=DynamicObject(obj="DSQAConfig", module="leaf_playground.zoo.dataset_qa.scene"),
        scene_schema=DSQASchema,
        is_terminal_impl=is_terminal,
        init_impl=init,
        new_methods={"get_data": get_data}
    )
)
