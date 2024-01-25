from sys import _getframe
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import create_model, BaseModel, Field, PositiveInt, PrivateAttr

from . import MetricDefinition
from .definitions import EnvVarDefinition, EnvVarConfig, MetricConfig, RoleDefinition, RoleConfig
from ..._config import _Config

_EnvVarName = str
_RoleName = str


class SceneDefinition(BaseModel):
    name: str = Field(default=...)
    description: str = Field(default=...)
    env_vars: List[EnvVarDefinition] = Field(default=...)
    roles: List[RoleDefinition] = Field(default=...)

    _log_exporters: List["leaf_playground.core.workers.logger.Logger"] = PrivateAttr(default=[])

    @property
    def roles_agent_num_range(self) -> Dict[str, Tuple[PositiveInt, Union[PositiveInt, Literal[-1]]]]:
        return {role.name: role.num_agents_range for role in self.roles}

    @property
    def log_exporters(self):
        return self._log_exporters

    def model_post_init(self, __context: Any) -> None:
        from leaf_playground.core.workers.logger import _KeptLogExporter

        if len(set([r.name for r in self.roles])) != len(self.roles):
            raise ValueError(f"roles should have unique names")
        if len(set([e.name for e in self.env_vars])) != len(self.env_vars):
            raise ValueError(f"env_vars should have unique names")

        for ext in ["json", "jsonl", "csv"]:
            self._log_exporters.append(_KeptLogExporter(extension=ext))

    def get_role_definition(self, role_name: str) -> RoleDefinition:
        for role in self.roles:
            if role.name == role_name:
                return role
        raise ValueError(f"role [{role_name}] not found")

    def get_env_var_definition(self, env_var_name: str) -> EnvVarDefinition:
        for env_var in self.env_vars:
            if env_var.name == env_var_name:
                return env_var
        raise ValueError(f"env_var [{env_var_name}] not found")

    def get_metric_definition(self, metric_belonged_chain: str) -> MetricDefinition:
        role_name, action_name, metric_name = metric_belonged_chain.split(".")
        return (
            self.get_role_definition(role_name).get_action_definition(action_name).get_metric_definition(metric_name)
        )

    def registry_log_exporters(self, log_exporters: List["leaf_playground.core.workers.logger.LogExporter"]):
        name2exporter = {f"{exporter.file_name}.{exporter.extension}": exporter for exporter in self.log_exporters}
        for exporter in log_exporters:
            name = f"{exporter.file_name}:{exporter.extension}"
            name2exporter[name] = exporter

        self._log_exporters = list(name2exporter.values())


class SceneEnvVarsConfig(_Config):
    @classmethod
    def create_config_model(cls, scene_definition: SceneDefinition) -> Optional[Type["SceneEnvVarsConfig"]]:
        model_name = "".join([each.capitalize() for each in scene_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        fields = {}
        for env_var in scene_definition.env_vars:
            fields[env_var.name] = (EnvVarConfig.create_config_model(env_var), Field(default=...))
        if not fields:
            return
        return create_model(__model_name=model_name, __module__=module, __base__=cls, **fields)

    def get_env_var_config(self, env_var_name: str) -> EnvVarConfig:
        return getattr(self, env_var_name)

    def initiate_env_vars(self) -> Dict[_EnvVarName, "leaf_playground.data.environment.EnvironmentVariable"]:
        return {f_name: getattr(self, f_name).initiate_env_var() for f_name in self.model_fields_set}


class SceneRolesConfig(_Config):
    @classmethod
    def create_config_model(cls, scene_definition: SceneDefinition) -> Type["SceneRolesConfig"]:
        model_name = "".join([each.capitalize() for each in scene_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        fields = {}
        for role in scene_definition.roles:
            fields[role.name] = (RoleConfig.create_config_model(role), Field(default=...))
        return create_model(__model_name=model_name, __module__=module, __base__=cls, **fields)

    def get_role_config(self, role_name: str) -> RoleConfig:
        return getattr(self, role_name)

    def initiate_agents(self) -> Dict[_RoleName, List["leaf_playground.core.scene_agent.SceneAgent"]]:
        return {f_name: getattr(self, f_name).initiate_agents() for f_name in self.model_fields_set}


class SceneConfig(_Config):
    env_vars_config: Optional[SceneEnvVarsConfig] = Field(default=...)
    roles_config: SceneRolesConfig = Field(default=...)

    @classmethod
    def create_config_model(
        cls,
        scene_definition: SceneDefinition,
        additional_config_fields: Optional[Dict[str, Tuple[Type, Field]]] = None,
    ) -> Type["SceneConfig"]:
        model_name = "".join([each.capitalize() for each in scene_definition.name.split("_")]) + cls.__name__
        module = _getframe(1).f_globals["__name__"]
        fields = {}
        env_vars_config_model = SceneEnvVarsConfig.create_config_model(scene_definition)
        if env_vars_config_model:
            fields["env_vars_config"] = (env_vars_config_model, Field(default=...))
        else:
            fields["env_vars_config"] = (Literal[None], Field(default=None))
        fields["roles_config"] = (SceneRolesConfig.create_config_model(scene_definition), Field(default=...))
        if additional_config_fields:
            fields.update(**additional_config_fields)

        return create_model(__model_name=model_name, __module__=module, __base__=cls, **fields)

    def get_env_var_config(self, env_var_name: str) -> EnvVarConfig:
        return self.env_vars_config.get_env_var_config(env_var_name)

    def get_role_config(self, role_name: str) -> RoleConfig:
        return self.roles_config.get_role_config(role_name)

    def get_metric_config(self, metric_belonged_chain: str) -> MetricConfig:
        role_name, action_name, metric_name = metric_belonged_chain.split(".")
        return self.get_role_config(role_name).get_action_config(action_name).get_metric_config(metric_name)

    def init_env_vars(self) -> Dict[_EnvVarName, "leaf_playground.data.environment.EnvironmentVariable"]:
        return self.env_vars_config.initiate_env_vars() if self.env_vars_config else {}

    def init_agents(self) -> Dict[_RoleName, List["leaf_playground.core.scene_agent.SceneAgent"]]:
        return self.roles_config.initiate_agents()


__all__ = ["SceneDefinition", "SceneEnvVarsConfig", "SceneRolesConfig", "SceneConfig"]
