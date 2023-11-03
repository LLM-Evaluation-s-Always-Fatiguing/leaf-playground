import random
from typing import List, Union

from pydantic import Field

from .dataset import load_hf_dataset, DatasetConfig
from ...scene.base import (
    Scene,
    SceneConfig,
    SceneSchema,
    SceneSubClsImplConfig,
    RoleSchema,
    RoleConfig,
    RoleDefinition,
    EnvironmentSchema,
    EnvironmentVarConfig,
    EnvironmentVariableDefinition
)
from ...utils.import_util import dynamically_import_obj, DynamicObject
from ...data.environment import ConstantEnvironmentVariable


ROLE_DEF_TEMPLATE = "RoleDefinition :: name={name}; description={description};"
ROLE_DEF_TEMPLATE_FIELDS = {"name", "description"}

ENV_VAR_DEF_TEMPLATE = "EnvironmentVariableDefinition :: name={name}; description={description};"
ENV_VAR_DEF_TEMPLATE_FIELDS = {"name", "description"}


class GaoKaoDatasetVariable(ConstantEnvironmentVariable):
    current_value: Union[List[str], int] = Field(default=...)


GaoKaoSceneSchema = SceneSchema(
    name="仿高考面试",
    description="以中国式高考为考核内容的面试场景，有一名考官和一名考生，考官负责提出问题（可能会给定候选答案），考生对考官的提问进行回答",
    role_schema=RoleSchema(
        num_participants=-1,
        definitions=[
            RoleDefinition(
                name="考官",
                description="面试的考核者，使用中国式高考的考题对考生进行面试",
                type=DynamicObject(
                    obj="Role",
                    module="leaf_playground.data.profile"
                ),
                role_num=1,
                template=ROLE_DEF_TEMPLATE,
                template_fields=ROLE_DEF_TEMPLATE_FIELDS
            ),
            RoleDefinition(
                name="考生",
                description="面试的被试者，对考官提出的问题（可能包含候选答案）进行回答",
                type=DynamicObject(
                    obj="Role",
                    module="leaf_playground.data.profile"
                ),
                role_num=-1,
                template=ROLE_DEF_TEMPLATE,
                template_fields=ROLE_DEF_TEMPLATE_FIELDS
            )
        ]
    ),
    environment_schema=EnvironmentSchema(
        definitions=[
            EnvironmentVariableDefinition(
                name="年份",
                description="题目所属的高考年份",
                type=DynamicObject(
                    obj="GaoKaoDatasetVariable",
                    module="leaf_playground.zoo.gaokao_bench.scene"
                ),
                template=ENV_VAR_DEF_TEMPLATE,
                template_fields=ENV_VAR_DEF_TEMPLATE_FIELDS
            ),
            EnvironmentVariableDefinition(
                name="题型",
                description="题目所属的学科和类型范围",
                type=DynamicObject(
                    obj="GaoKaoDatasetVariable",
                    module="leaf_playground.zoo.gaokao_bench.scene"
                ),
                template=ENV_VAR_DEF_TEMPLATE,
                template_fields=ENV_VAR_DEF_TEMPLATE_FIELDS
            ),
            EnvironmentVariableDefinition(
                name="题数",
                description="题目数量",
                type=DynamicObject(
                    obj="GaoKaoDatasetVariable",
                    module="leaf_playground.zoo.gaokao_bench.scene"
                ),
                template=ENV_VAR_DEF_TEMPLATE,
                template_fields=ENV_VAR_DEF_TEMPLATE_FIELDS
            )
        ]
    )
)


class GaoKaoEnvVarConfig(EnvironmentVarConfig):
    env_var_obj: DynamicObject = DynamicObject(
        obj="GaoKaoDatasetVariable",
        module="leaf_playground.zoo.gaokao_bench.scene"
    )


class GaoKaoSceneConfig(SceneConfig):
    env_var_configs: List[GaoKaoEnvVarConfig] = Field(default=...)


def init(self: Scene):
    ds_config = DatasetConfig(
        years=self.get_env_var("年份").current_value,
        types=self.get_env_var("题型").current_value
    )
    data = load_hf_dataset(ds_config).to_list()
    self._data = random.sample(data, min(len(data), self.get_env_var("题数").current_value))


def is_terminal(self: Scene) -> bool:
    if not self._data:
        return True
    return False


def get_data(self: Scene) -> dict:
    data = self._data.pop(0)
    return data


GaoKaoScene = Scene.implement_sub_cls(
    config=SceneSubClsImplConfig(
        cls_name="GaoKaoScene",
        config_obj=DynamicObject(obj="GaoKaoSceneConfig", module="leaf_playground.zoo.gaokao_bench.scene"),
        scene_schema=GaoKaoSceneSchema,
        is_terminal_impl=is_terminal,
        init_impl=init,
        new_methods={"get_data": get_data}
    )
)
