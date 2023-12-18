import asyncio

from leaf_playground.core.scene_engine import (
    SceneEngine, SceneObjConfig, MetricEvaluatorObjConfig, MetricEvaluatorObjsConfig
)
from leaf_playground.zoo_new.general_mcq_examine import *


def init_scene_engine():
    scene_engine = SceneEngine(
        scene_config=SceneObjConfig(
            scene_config_data={
                "roles_config": {
                    "examiner": {
                        "actions_config": {},
                    },
                    "examinee": {
                        "actions_config": {
                            "answer_question": {"metrics_config": {"accurate": {"enable": True}}}
                        },
                        "agents_config": [
                            {
                                "config_data": {
                                    "profile": {"name": "William"},
                                    "ai_backend_config": {"model": "gpt-3.5-turbo"}
                                },
                                "obj_for_import": OpenAIBasicExaminee.obj_for_import.model_dump(mode="json", by_alias=True)
                            }
                        ]
                    }
                },
                "dataset_config": {
                    "path": "AsakusaRinne/gaokao_bench",
                    "name": "2010-2022_History_MCQs",
                    "split": "dev",
                    "question_column": "question",
                    "golden_answer_column": "answer",
                    "num_questions": 3
                }
            },
            scene_obj=GeneralMCQExamineScene.obj_for_import
        ),
        evaluators_config=MetricEvaluatorObjsConfig(
            evaluators=[
                MetricEvaluatorObjConfig(
                    evaluator_config_data={},
                    evaluator_obj=ExamineeAnswerEvaluator.obj_for_import
                )
            ]
        )
    )
    return scene_engine


def display_log(log: GeneralMCQExamineScene.log_body_class):
    narrator = log.narrator
    sender = log.response.sender.name
    sender_role = log.response.sender.role.name
    content = log.response.content.text.strip()
    print(f"({narrator})\n", flush=True)
    print(f"{sender}({sender_role}): {content}\n", flush=True)
    print(f"ground_truth: {log.ground_truth}\n", flush=True)


async def run():
    scene_engine = init_scene_engine()
    await asyncio.gather(scene_engine.a_run(), scene_engine.stream_logs(display_log))
    scene_engine.save_dir = f"output/rag_qa_result-{scene_engine.id.hex}"
    scene_engine.save()


if __name__ == "__main__":
    asyncio.run(run())
