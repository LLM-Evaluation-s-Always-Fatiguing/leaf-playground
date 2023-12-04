import asyncio
from datetime import datetime

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
from leaf_playground.zoo.rag_qa.dataset_utils import DatasetConfig
from leaf_playground.zoo.rag_qa.scene import (
    RagQaScene,
    RagQaSceneConfig,
)
from leaf_playground.zoo.rag_qa.scene_agent import ExamineeAnswer, ExaminerQuestion
from leaf_playground.zoo.rag_qa.scene_info import envs_config_model

agent_obj = RagQaScene.get_dynamic_agent_classes()[0]
evaluator_obj = RagQaScene.get_evaluator_classes()[0]
scene_info_obj = RagQaScene.get_scene_info_class().obj_for_import

scene_config = RagQaSceneConfig(
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
                        ai_backend_config=OpenAIBackendConfig(model="gpt-3.5-turbo-16k"),
                        # ai_backend_config=OpenAIBackendConfig(model="gpt-4-1106-preview"),
                        chart_major_color="#4513de",
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
                    "evaluator_config_data": {
                        "activate_metrics": ["answer_correctness", "answer_relevancy", "context_precision"]
                    },
                    "evaluator_obj": evaluator_obj.model_dump(by_alias=True)
                }
            )
        ]
    ),
    dataset_config=DatasetConfig(
        **{
            "path": "explodinggradients/fiqa",
            "name": "ragas_eval",
            "split": "baseline",
            "question_column": "question",
            "golden_answer_column": "answer",
            "ground_truth_column": "ground_truths",
            "num_questions": 1
        }
    ),
    activate_metrics=["answer_correctness", "answer_relevancy", "context_precision"],
    debug_mode=True
)


def display_log(log: RagQaScene.log_body_class):
    narrator = log.narrator
    sender = log.response.sender.name
    sender_role = log.response.sender.role.name
    content = log.response.content.text.strip() if isinstance(log.response, ExaminerQuestion) else \
        log.response.content.data['answer']
    contexts = log.response.content.data['contexts'] if isinstance(log.response, ExamineeAnswer) else []
    print(f"({narrator})\n", flush=True)
    print(f"{sender}({sender_role}): {content}\n", flush=True)
    print(f"ground_truth: {log.ground_truth}\n", flush=True)
    print(f"contexts: {contexts}\n", flush=True)


async def run_scene():
    scene = RagQaScene.from_config(config=scene_config)
    await asyncio.gather(scene.a_start(), scene.stream_logs(display_log))
    scene.save_dir = f"output/rag_qa_result-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    scene.save()


if __name__ == "__main__":
    asyncio.run(run_scene())
