import json
import os
import requests
import shutil
import subprocess
import sys
import zipfile
from abc import ABC
from collections import defaultdict
from packaging import version
from pathlib import Path
from typing import List, Optional, Type
from typing_extensions import Annotated
from urllib.request import urlopen

import typer
from cookiecutter.main import cookiecutter
from rich.progress import wrap_file

from leaf_playground import __version__ as leaf_version
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_agent import SceneAgent
from leaf_playground.core.workers import MetricEvaluator
from leaf_playground.core.workers.chart import Chart
from leaf_playground.utils.import_util import relevantly_find_subclasses

app = typer.Typer(name="leaf-playground-cli")
template_dir = os.path.join(os.path.dirname(__file__), "templates")


@app.command(name="new-project")
def create_new_project(
        name: Annotated[str, typer.Argument(metavar="project_name")]
):
    project_name = name.lower().replace(" ", "_").replace("-", "_")
    cookiecutter(
        template=template_dir,
        extra_context={
            "project_name": project_name,
            "project_name_camel_case": "".join([each.capitalize() for each in project_name.split("_")]),
            "leaf_version": leaf_version
        },
        no_input=True
    )

    print(f"project [{name}] created.")
    raise typer.Exit()


@app.command(name="publish")
def publish_project(
        target: Annotated[str, typer.Argument(metavar="target_dir")],
        version_str: Annotated[str, typer.Option("--version", "-v")] = "0.1.0",
):
    dot_leaf_dir = os.path.join(target, ".leaf")

    if not os.path.exists(dot_leaf_dir):
        raise typer.BadParameter("not a leaf playground project.")

    try:
        version.parse(version_str)
    except version.InvalidVersion:
        raise typer.BadParameter(f"expect a valid version string, got {version_str}")

    with open(os.path.join(dot_leaf_dir, "project_config.json"), "r", encoding="utf-8") as f:
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
    chart_classes: List[Type[Chart]] = relevantly_find_subclasses(
        root_path=os.path.join(pkg_root, "charts"),
        prefix=f"{project_name}.charts",
        base_class=Chart
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
        ] if evaluator_classes else None,
        "charts_metadata": [
            chart_cls.get_metadata().model_dump(mode="json", by_alias=True) for chart_cls in chart_classes
        ] if chart_classes else None
    }

    with open(os.path.join(dot_leaf_dir, "project_config.json"), "w", encoding="utf-8") as f:
        json.dump(project_config, f, indent=4, ensure_ascii=False)

    app_py_file_path = os.path.join(
        template_dir,
        "{{cookiecutter.project_name}}",
        ".leaf",
        "app.py"
    )
    shutil.copy(app_py_file_path, os.path.join(dot_leaf_dir, "app.py"))

    print(f"publish new version [{version_str}]")
    raise typer.Exit()


# TODO: change to official site after open sourced
__web_ui_release_site__ = "https://github.com/AIEPhoenix/aie-docusaurus-template/releases/download/v1.0.0/"
__web_ui_download_url__ = __web_ui_release_site__ + "webui-v0.5.0.zip"
__web_ui_hash_url__ = __web_ui_release_site__ + "webui-v0.5.0.zip.sha256"


def download_web_ui() -> str:
    leaf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".leaf_workspace")
    tmp_dir = os.path.join(leaf_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    web_ui_dir = os.path.join(leaf_dir, "web_ui")
    os.makedirs(web_ui_dir, exist_ok=True)

    web_ui_file_name = os.path.split(__web_ui_download_url__)[-1]
    web_ui_hash_file_name = os.path.split(__web_ui_hash_url__)[-1]

    web_ui_save_dir = os.path.join(web_ui_dir, os.path.splitext(web_ui_file_name)[0])
    web_ui_file_save_path = os.path.join(tmp_dir, web_ui_file_name)

    # check hash is the same
    remote_hash = requests.get(__web_ui_hash_url__).content.decode(encoding="utf-8")
    local_hash_save_path = os.path.join(web_ui_dir, web_ui_hash_file_name)
    if os.path.exists(local_hash_save_path):
        with open(local_hash_save_path, "r", encoding="utf-8") as f:
            local_hash = f.read().strip()
        if remote_hash == local_hash:
            return web_ui_save_dir
    with open(local_hash_save_path, "w", encoding="utf-8") as f:
        f.write(remote_hash)

    # (re-)download web ui
    if os.path.exists(web_ui_save_dir):
        shutil.rmtree(web_ui_save_dir)
    response = urlopen(__web_ui_download_url__)
    size = int(response.headers["Content-Length"])
    with open(web_ui_file_save_path, "wb") as local_file:
        with wrap_file(response, size, description="download web ui...") as remote_file:
            for chunk in remote_file:
                local_file.write(chunk)
    with zipfile.ZipFile(web_ui_file_save_path, 'r') as zip_ref:
        zip_ref.extractall(web_ui_save_dir)
    os.remove(web_ui_file_save_path)

    return web_ui_save_dir


@app.command(name="start-server")
def start_server(
    zoo_dir: Annotated[str, typer.Option("--zoo")] = os.getcwd(),
    port: Annotated[int, typer.Option("--port", "-p")] = 8000,
    dev_dir: Annotated[Optional[str], typer.Option("--dev_dir")] = None,
    web_ui_dir: Annotated[Optional[str], typer.Option("--web_ui_dir")] = None
):
    if web_ui_dir and not os.path.isdir(web_ui_dir):
        raise typer.BadParameter("value of argument '--web_ui' must be a local existed directory")
    if not web_ui_dir:
        web_ui_dir = download_web_ui()
    subprocess.Popen(f"node {os.path.join(web_ui_dir, 'server.js')}".split())

    if dev_dir:
        dev_dir = os.path.abspath(dev_dir)
        sys.path.insert(0, dev_dir)

    from .service import init_service, ServiceConfig

    if not os.path.exists(zoo_dir):
        raise typer.BadParameter(f"zoo [{zoo_dir}] not exist.")

    init_service(config=ServiceConfig(zoo_dir=zoo_dir, port=port))

    import uvicorn

    uvicorn.run(
        "leaf_playground_cli.service:app",
        port=port,
        host="127.0.0.1",
        reload=bool(dev_dir),
        reload_dirs=dev_dir
    )


if __name__ == "__main__":
    app()
