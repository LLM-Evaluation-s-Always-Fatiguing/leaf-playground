from enum import Enum
from pydantic import Field

from leaf_playground.data.environment import ConstantEnvironmentVariable
from leaf_playground.core.scene_info import (
    RoleDefinition,
    EnvVarDefinition,
    SceneMetaData,
    SceneInfoConfigBase,
    SceneInfo
)
from leaf_playground.utils.import_util import DynamicObject


class PlayerRoles(Enum):
    CIVILIAN = "civilian"
    SPY = "spy"
    BLANK_SLATE = "blank_slate"


class PlayerStatus(Enum):
    ALIVE = "alive"
    ELIMINATED = "eliminated"


class KeyModalities(Enum):
    TEXT = "text"
    # IMAGE = "image"
    # AUDIO = "audio"


class KeyModalityEnvVar(ConstantEnvironmentVariable):
    current_value: KeyModalities = Field(default=KeyModalities.TEXT)


class NumRoundsEnvVar(ConstantEnvironmentVariable):
    current_value: int = Field(default=1)


class HasBlankSlateEnvVar(ConstantEnvironmentVariable):
    current_value: bool = Field(default=False)


class WhoIsTheSpySceneMetaData(SceneMetaData):
    def get_max_dynamic_agents_num(self) -> int:
        return 9

    def get_min_dynamic_agents_num(self) -> int:
        return 4  # when only has 4 players, there is no blank slate


who_is_the_spy_scene_metadata = WhoIsTheSpySceneMetaData(
    name="WhoIsTheSpy",
    description="A scene that simulates the Who is the Spy game.",
    role_definitions=[
        RoleDefinition(
            name="player",
            description="the one that plays the game",
            num_agents=-1,
            is_static=False,
        ),
        RoleDefinition(
            name="moderator",
            description="the one that moderate the game",
            num_agents=1,
            is_static=True,
            agent_type=DynamicObject(obj="Moderator", module="leaf_playground.zoo.who_is_the_spy.scene_agent")
        ),
    ],
    env_definitions=[
        EnvVarDefinition(
            name="key_modality",
            description="the modality of the key, i.e: text, image, audio, etc.",
            type=DynamicObject(obj="KeyModalityEnvVar", module="leaf_playground.zoo.who_is_the_spy.scene_info")
        ),
        EnvVarDefinition(
            name="num_rounds",
            description="num rounds to play",
            type=DynamicObject(obj="NumRoundsEnvVar", module="leaf_playground.zoo.who_is_the_spy.scene_info")
        ),
        EnvVarDefinition(
            name="has_blank_slate",
            description="whether the game has blank slates",
            type=DynamicObject(obj="HasBlankSlateEnvVar", module="leaf_playground.zoo.who_is_the_spy.scene_info")
        ),
    ]
)

(
    WhoIsTheSpySceneInfoConfig,
    roles_config_model,
    envs_config_model,
    roles_cls,
    envs_cls
) = SceneInfoConfigBase.create_subclass(
    who_is_the_spy_scene_metadata
)


class WhoIsTheSpySceneInfo(SceneInfo):
    config_obj = WhoIsTheSpySceneInfoConfig
    config: config_obj

    obj_for_import = DynamicObject(obj="WhoIsTheSpySceneInfo", module="leaf_playground.zoo.who_is_the_spy.scene_info")

    def __init__(self, config: config_obj):
        super().__init__(config=config)


__all__ = [
    "PlayerRoles",
    "PlayerStatus",
    "KeyModalities",
    "who_is_the_spy_scene_metadata",
    "WhoIsTheSpySceneInfoConfig",
    "WhoIsTheSpySceneInfo"
]
