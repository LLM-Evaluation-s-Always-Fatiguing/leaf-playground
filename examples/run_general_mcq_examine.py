import asyncio

from leaf_playground.ai_backend.openai import OpenAIBackendConfig
from leaf_playground.core.scene import (
    SceneAgentsObjConfig,
    SceneAgentObjConfig,
    SceneEvaluatorObjConfig,
    SceneEvaluatorsObjConfig,
    SceneInfoObjConfig
)
from leaf_playground.data.profile import Profile
from leaf_playground.utils.import_util import dynamically_import_obj
from leaf_playground.zoo.general_mcq_examine.dataset_utils import DatasetConfig
from leaf_playground.zoo.general_mcq_examine.scene import (
    GeneralMCQExamineScene,
    GeneralMCQExamineSceneConfig,
)
from leaf_playground.zoo.general_mcq_examine.scene_info import envs_config_model

agent_obj = GeneralMCQExamineScene.get_dynamic_agent_classes()[0]
evaluator_obj = GeneralMCQExamineScene.get_evaluator_classes()[0]
scene_info_obj = GeneralMCQExamineScene.get_scene_info_class().obj_for_import

scene_config = GeneralMCQExamineSceneConfig(
    scene_info=SceneInfoObjConfig(
        **{
            "scene_info_config_data": {"environments": envs_config_model()},
            "scene_info_obj": scene_info_obj.model_dump(by_alias=True)
        }
    ),
    scene_agents=SceneAgentsObjConfig(
        agents=[
            SceneAgentObjConfig(
                **{
                    "agent_config_data": dynamically_import_obj(agent_obj).config_obj(
                        profile=Profile(name="James"),
                        ai_backend_config=OpenAIBackendConfig(model="gpt-4-1106-preview"),
                    ).model_dump(by_alias=True),
                    "agent_obj": agent_obj.model_dump(by_alias=True)
                }
            ),
            SceneAgentObjConfig(
                **{
                    "agent_config_data": dynamically_import_obj(agent_obj).config_obj(
                        profile=Profile(name="William"),
                        ai_backend_config=OpenAIBackendConfig(model="gpt-3.5-turbo-instruct"),
                    ).model_dump(by_alias=True),
                    "agent_obj": agent_obj.model_dump(by_alias=True)
                }
            )
        ]
    ),
    scene_evaluators=SceneEvaluatorsObjConfig(
        evaluators=[
            SceneEvaluatorObjConfig(
                **{
                    "evaluator_config_data": {},
                    "evaluator_obj": evaluator_obj.model_dump(by_alias=True)
                }
            )
        ]
    ),
    dataset_config=DatasetConfig(
        **{
            "path": "AsakusaRinne/gaokao_bench",
            "name": "2010-2022_History_MCQs",
            "split": "dev",
            "question_column": "question",
            "golden_answer_column": "answer",
            "num_questions": 10
        }
    )
)


def display_log(log: GeneralMCQExamineScene.log_body_class):
    narrator = log.narrator
    sender = log.response.sender.name
    sender_role = log.response.sender.role.name
    content = log.response.content.text.strip()
    print(f"({narrator})\n", flush=True)
    print(f"{sender}({sender_role}): {content}\n", flush=True)


async def run_scene():
    scene = GeneralMCQExamineScene.from_config(config=scene_config)
    await asyncio.gather(scene.a_start(), scene.stream_logs(display_log))
    scene.save("output/general_mcq_examine_result")


if __name__ == "__main__":
    asyncio.run(run_scene())
