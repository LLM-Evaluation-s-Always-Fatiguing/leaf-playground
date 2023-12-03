from leaf_playground.core.scene_info import (
    RoleDefinition,
    SceneMetaData,
    SceneInfoConfigBase,
    SceneInfo
)
from leaf_playground.utils.import_util import DynamicObject

rag_qa_scene_metadata = SceneMetaData(
    name="RAG QA Examine",
    description="Retrieval Augmented Generation Question Answering Examine Scene. The evaluator powered by ragas.",
    role_definitions=[
        RoleDefinition(
            name="examiner",
            description="the one that participants in a rag based qa examine to monitor the examinees",
            num_agents=1,
            is_static=True,
            type=DynamicObject(obj="Role", module="leaf_playground.data.profile"),
            agent_type=DynamicObject(obj="Examiner", module="leaf_playground.zoo.rag_qa.scene_agent")
        ),
        RoleDefinition(
            name="examinee",
            description="the one that participants in a rag based qa examine to answer questions",
            num_agents=-1,
            type=DynamicObject(obj="Role", module="leaf_playground.data.profile"),
            is_static=False
        )
    ],
    env_definitions=[]
)

(
    RagQaSceneInfoConfig,
    roles_config_model,
    envs_config_model,
    roles_cls,
    envs_cls
) = SceneInfoConfigBase.create_subclass(
    rag_qa_scene_metadata
)


class RagQaSceneInfo(SceneInfo):
    config_obj = RagQaSceneInfoConfig
    config: config_obj

    obj_for_import = DynamicObject(
        obj="RagQaSceneInfo", module="leaf_playground.zoo.rag_qa.scene_info"
    )

    def __init__(self, config: config_obj):
        super().__init__(config=config)


__all__ = [
    "rag_qa_scene_metadata",
    "RagQaSceneInfoConfig",
    "RagQaSceneInfo"
]
