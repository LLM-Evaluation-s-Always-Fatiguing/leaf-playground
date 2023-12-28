from leaf_playground.core.workers import Logger
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_definition import SceneConfig
from leaf_playground.data.log_body import ActionLogBody

from .scene_definition import *


class {{ cookiecutter.project_name_camel_case }}SceneLogBody(ActionLogBody):
    pass


{{ cookiecutter.project_name_camel_case }}SceneConfig = SceneConfig.create_config_model(
    SCENE_DEFINITION,
)


class {{ cookiecutter.project_name_camel_case }}Scene(
    Scene,
    scene_definition=SCENE_DEFINITION,
    log_body_class={{ cookiecutter.project_name_camel_case }}SceneLogBody
):
    config_cls = {{ cookiecutter.project_name_camel_case }}SceneConfig
    config: config_cls

    def __init__(self, config: config_cls, logger: Logger):
        super().__init__(config=config, logger=logger)
        # TODO: additional initialization logic here

    async def _run(self):
        # TODO: scene flow here
        pass


__all__ = [
    "{{ cookiecutter.project_name_camel_case }}SceneConfig",
    "{{ cookiecutter.project_name_camel_case }}Scene"
]
