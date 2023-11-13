from argparse import ArgumentParser

from leaf_playground.zoo.dataset_qa.engine import DSQAEngine, DSQAEngineConfig

parser = ArgumentParser()
parser.add_argument("--openai_api_key", type=str)
args = parser.parse_args()

config = DSQAEngineConfig(
    **{
        "agents_obj": [
            {
                "config_data": {"profile_data": {"name": "Jane"}},
                "obj": {"obj": "Examiner", "module": "leaf_playground.zoo.dataset_qa.agent"}
            },
            {
                "config_data": {
                    "profile_data": {"name": "William"},
                    "backend_config_data": {
                        "model": "gpt-3.5-turbo-instruct",
                        "api_key": args.openai_api_key,
                        "completion_kwargs": {"max_tokens": 1}
                    },
                    "backend_obj": {"obj": "OpenAIBackend", "module": "leaf_playground.llm_backend.openai"},
                    "answer_prefix": "My answer is option"
                },
                "obj": {"obj": "Examinee", "module": "leaf_playground.zoo.dataset_qa.agent"}
            },
        ],
        "scene_obj": {
            "config_data": {
                "role_configs": [
                    {
                        "role_data": {
                            "name": "examinee", "description": "the one that answer the question sent by a examiner"
                        }
                    },
                    {
                        "role_data": {
                            "name": "examiner", "description": "the one that select a question from a given dataset and send to all examinees"
                        }
                    },
                ],
                "env_var_configs": [],
                "dataset_config": {
                    "path": "AsakusaRinne/gaokao_bench",
                    "name": "2010-2022_History_MCQs",
                    "split": "dev",
                    "question_column": "question",
                    "golden_answer_column": "answer"
                }
            },
            "obj": {"obj": "DSQA", "module": "leaf_playground.zoo.dataset_qa.scene"}
        },
    }
)

if __name__ == "__main__":
    engine = DSQAEngine(config=config)
    engine.start()
    engine.export_logs("./dsqa_log.jsonl")
    engine.export_examine_results("./dsqa_res.jsonl")
