from argparse import ArgumentParser

from leaf_playground.zoo.gaokao_bench.engine import GaoKaoBench, GaoKaoBenchConfig

parser = ArgumentParser()
parser.add_argument("--openai_api_key", type=str)
parser.add_argument("--hf_model_name_or_path", type=str, default="baichuan-inc/Baichuan2-7B-Base")
parser.add_argument("--trust_remote_code", action="store_true")
args = parser.parse_args()

config = GaoKaoBenchConfig(
    **{
        "agents_obj": [
            {
                "config_data": {
                    "profile_data": {"name": "William"},
                    "backend_config_data": {
                        "model": "gpt-3.5-turbo-instruct",
                        "api_key": args.openai_api_key,
                        "completion_hyper_params": {"max_tokens": 2}
                    },
                    "backend_obj": {"obj": "OpenAIBackend", "module": "leaf_playground.llm_backend.openai"}
                },
                "obj": {"obj": "Student", "module": "leaf_playground.zoo.gaokao_bench.agent"}
            },
            {
                "config_data": {
                    "profile_data": {"name": "Jordan"},
                    "backend_config_data": {
                        "model_name_or_path": args.hf_model_name_or_path,
                        "trust_remote_code": args.trust_remote_code,
                        "device": 0,
                        "generation_config": {"max_new_tokens": 2}
                    },
                    "backend_obj": {"obj": "HuggingFaceBackend", "module": "leaf_playground.llm_backend.huggingface"}
                },
                "obj": {"obj": "Student", "module": "leaf_playground.zoo.gaokao_bench.agent"}
            },
            {
                "config_data": {
                    "profile_data": {"name": "Jane"},
                },
                "obj": {"obj": "Teacher", "module": "leaf_playground.zoo.gaokao_bench.agent"}
            }
        ],
        "scene_obj": {
            "config_data": {
                "role_configs": [
                    {
                        "role_data": {"name": "考官", "description": "面试的考核者，使用中国式高考的考题对考生进行面试"}
                    },
                    {
                        "role_data": {
                            "name": "考生", "description": "面试的被试者，对考官提出的问题（可能包含候选答案）进行回答"
                        }
                    }
                ],
                "env_var_configs": [
                    {
                        "env_var_data": {
                            "name": "年份", "description": "题目所属的高考年份",
                            "current_value": ["2013", "2014", "2015"]
                        },
                    },
                    {
                        "env_var_data": {
                            "name": "题型", "description": "题目所属的学科和类型范围",
                            "current_value": ["2010-2022_History_MCQs", "2010-2022_Physics_MCQs"]
                        }
                    },
                    {
                        "env_var_data": {"name": "题数", "description": "题目数量", "current_value": 10}
                    }
                ]
            },
            "obj": {"obj": "GaoKaoScene", "module": "leaf_playground.zoo.gaokao_bench.scene"}
        },
    }
)

if __name__ == "__main__":
    engine = GaoKaoBench(config=config)
    engine.start()
    engine.export_logs("./gaokao_bench_simulation_log.jsonl")
