from leaf_playground.core.scene_info import (
    RoleDefinition,
    EnvVarDefinition,
    SceneMetaData,
    SceneInfoConfigBase,
    SceneInfo
)
from leaf_playground.utils.import_util import DynamicObject

general_mcq_examine_scene_metadata = SceneMetaData(
    name="GeneralMCQExamine",
    description="A general examine scene that uses dataset from huggingface hub to test agents.",
    role_definitions=[
        RoleDefinition(
            name="examiner",
            description="the one that participants in an multiple choices question examine to monitor the examinees",
            num_agents=1,
            is_static=True,
            type=DynamicObject(obj="Role", module="leaf_playground.data.profile"),
            agent_type=DynamicObject(obj="Examiner", module="leaf_playground.zoo.general_mcq_examine.scene_agent")
        ),
        RoleDefinition(
            name="examinee",
            description="the one that participants in an multiple choices question examine to answer questions",
            num_agents=-1,
            type=DynamicObject(obj="Role", module="leaf_playground.data.profile"),
            is_static=False
        )
    ],
    env_definitions=[]
)

(
    GeneralMCQExamineSceneInfoConfig,
    roles_config_model,
    envs_config_model,
    roles_cls,
    envs_cls
) = SceneInfoConfigBase.create_subclass(
    general_mcq_examine_scene_metadata
)


class GeneralMCQExamineSceneInfo(SceneInfo):
    config_obj = GeneralMCQExamineSceneInfoConfig
    config: config_obj

    obj_for_import = DynamicObject(
        obj="GeneralMCQExamineSceneInfo", module="leaf_playground.zoo.general_mcq_examine.scene_info"
    )

    def __init__(self, config: config_obj):
        super().__init__(config=config)


__all__ = [
    "general_mcq_examine_scene_metadata",
    "GeneralMCQExamineSceneInfoConfig",
    "GeneralMCQExamineSceneInfo"
]
