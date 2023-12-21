import json
import os
import sys
from abc import ABC
from collections import defaultdict
from packaging import version
from pathlib import Path
from typing import List, Type
from typing_extensions import Annotated

import typer

from .service import start_service, ServiceConfig
from leaf_playground import __version__ as leaf_version
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_agent import SceneAgent
from leaf_playground.core.workers import MetricEvaluator
from leaf_playground.utils.import_util import relevantly_find_subclasses


app = typer.Typer(name="leaf-playground-cli")


@app.command(name="new_project")
def create_new_project(
    name: Annotated[str, typer.Argument(metavar="project_name")]
):
    project_dir = os.path.join("./",  name)

    if os.path.exists(project_dir):
        print(f"project [{name}] already existed.")
        raise typer.Exit()

    os.makedirs(project_dir)
    os.makedirs(os.path.join(project_dir, name))
    with open(os.path.join(project_dir, name, "__init__.py"), "w", encoding="utf-8") as f:
        f.write("")
    os.makedirs(os.path.join(project_dir, ".leaf"))
    with open(os.path.join(project_dir, ".leaf", "project_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"name": name, "version": "0.1.0", "metadata": {}, "leaf_version": leaf_version},
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"project [{name}] created.")
    raise typer.Exit()


@app.command(name="publish")
def publish_project(
    target: Annotated[str, typer.Argument(metavar="target_dir")],
    version_str: Annotated[str, typer.Option("--version", "-v")] = "0.1.0",
):
    if not os.path.exists(os.path.join(target, ".leaf")):
        raise typer.BadParameter("not a leaf playground project.")

    try:
        version.parse(version_str)
    except version.InvalidVersion:
        raise typer.BadParameter(f"expect a valid version string, got {version_str}")

    with open(os.path.join(target, ".leaf", "project_config.json"), "r", encoding="utf-8") as f:
        project_config = json.load(f)
        old_version_str = project_config["version"]
        project_name = project_config["name"]
    if version.parse(version_str) < version.parse(old_version_str):
        raise typer.BadParameter(f"expect version string greater or equal to {old_version_str}, got {version_str}")

    # set new version
    project_config["version"] = version_str
    project_config["leaf_version"] = leaf_version

    # get and set metadata
    pkg_root = Path(os.path.join(target, project_name))
    sys.path.insert(0, pkg_root.parent.as_posix())
    scene_class: Type[Scene] = relevantly_find_subclasses(
        root_path=pkg_root.as_posix(),
        prefix=project_name,
        base_class=Scene
    )[0]
    agent_classes: List[Type[SceneAgent]] = relevantly_find_subclasses(
        root_path=os.path.join(pkg_root, "agents"),
        prefix=f"{project_name}.agents",
        base_class=SceneAgent
    )
    evaluator_classes: List[Type[MetricEvaluator]] = relevantly_find_subclasses(
        root_path=os.path.join(pkg_root, "metric_evaluators"),
        prefix=f"{project_name}.metric_evaluators",
        base_class=MetricEvaluator
    )

    agents_metadata = defaultdict(list)
    for agent_cls in agent_classes:
        if ABC in agent_cls.__bases__:
            continue
        agents_metadata[agent_cls.role_definition.name].append(
            agent_cls.get_metadata().model_dump(mode="json", by_alias=True)
        )

    project_config["metadata"] = {
        "scene_metadata": scene_class.get_metadata().model_dump(mode="json", by_alias=True),
        "agents_metadata": agents_metadata,
        "evaluators_metadata": [
            evaluator_cls.get_metadata().model_dump(mode="json", by_alias=True) for evaluator_cls in evaluator_classes
        ] if evaluator_classes else None
    }

    with open(os.path.join(target, ".leaf", "project_config.json"), "w", encoding="utf-8") as f:
        json.dump(project_config, f, indent=4, ensure_ascii=False)

    print(f"publish new version [{version_str}]")
    raise typer.Exit()


@app.command(name="start-server")
def start_server(
    zoo_dir: Annotated[str, typer.Option("--zoo")] = os.getcwd(),
    results_dir: Annotated[str, typer.Option("--results")] = os.path.join(os.getcwd(), "results"),
    port: Annotated[int, typer.Option("--port", "-p")] = 8000
):
    if not os.path.exists(zoo_dir):
        raise typer.BadParameter(f"zoo [{zoo_dir}] not exist.")
    os.makedirs(results_dir, exist_ok=True)

    start_service(config=ServiceConfig(zoo_dir=zoo_dir, result_dir=results_dir, port=port))


if __name__ == "__main__":
    app()
