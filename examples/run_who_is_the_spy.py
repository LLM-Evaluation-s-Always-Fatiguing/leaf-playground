import asyncio

from leaf_playground.zoo.who_is_the_spy.scene import WhoIsTheSpyScene, WhoIsTheSpySceneConfig

agent_obj = WhoIsTheSpyScene.get_dynamic_agent_classes()[0]
scene_info_obj = WhoIsTheSpyScene.get_scene_info_class().obj_for_import

scene_config = WhoIsTheSpySceneConfig(
    **{
        "debug_mode": True,
        "scene_info": {
            "scene_info_config_data": {
                "environments": {
                    "key_modality": {"current": "text"},
                    "num_rounds": {"current": 1},
                    "has_blank_slate": {"current": False},
                }
            },
            "scene_info_obj": scene_info_obj.model_dump(by_alias=True)
        },
        "scene_agents": {
            "agents": [
                {
                    "agent_config_data": {
                        "profile": {"name": "William"},
                        "ai_backend_config": {"model": "gpt-4"}
                    },
                    "agent_obj": agent_obj.model_dump(by_alias=True)
                },
                {
                    "agent_config_data": {
                        "profile": {"name": "James"},
                        "ai_backend_config": {"model": "gpt-4-0613"}
                    },
                    "agent_obj": agent_obj.model_dump(by_alias=True)
                },
                {
                    "agent_config_data": {
                        "profile": {"name": "Jordan"},
                        "ai_backend_config": {"model": "gpt-3.5-turbo-16k"},
                        "context_max_tokens": 16000
                    },
                    "agent_obj": agent_obj.model_dump(by_alias=True)
                },
                {
                    "agent_config_data": {
                        "profile": {"name": "Alex"},
                        "ai_backend_config": {"model": "gpt-3.5-turbo"}
                    },
                    "agent_obj": agent_obj.model_dump(by_alias=True)
                },
            ]
        },
        "scene_evaluators": {
            "evaluators": []
        },
    }
)


def display_log(log: WhoIsTheSpyScene.log_body_class):
    sender = log.response.sender.name
    sender_role = log.response.sender.role.name
    content = log.response.content.text.strip()
    print(f"{sender}({sender_role}): {content}\n", flush=True)


async def run_scene():
    scene = WhoIsTheSpyScene.from_config(config=scene_config)
    await asyncio.gather(scene.a_start(), scene.stream_logs(display_log))
    scene.save("who_is_the_spy_result")


if __name__ == "__main__":
    asyncio.run(run_scene())
