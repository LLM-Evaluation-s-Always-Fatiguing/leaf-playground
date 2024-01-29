import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from abc import ABC
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Type
from urllib.request import urlopen
from uuid import uuid4

import click
import requests
import typer
from cookiecutter.main import cookiecutter
from packaging import version
from rich.progress import wrap_file
from typing_extensions import Annotated

from leaf_playground import __version__ as leaf_version
from leaf_playground.core.scene import Scene
from leaf_playground.core.scene_agent import SceneAgent
from leaf_playground.core.workers import MetricEvaluator
from leaf_playground.core.workers.chart import Chart
from leaf_playground.utils.import_util import relevantly_find_subclasses

app = typer.Typer(name="leaf-playground-cli")
template_dir = os.path.join(os.path.dirname(__file__), "templates")


@app.command(name="new-project", help="initialize a new leaf-playground scenario simulation project from template.")
def create_new_project(name: Annotated[str, typer.Argument(metavar="project_name")]):
    project_name = name.lower().replace(" ", "_").replace("-", "_")
    project_id = project_name + "_" + uuid4().hex[:8]
    created = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    cookiecutter(
        template=template_dir,
        extra_context={
            "project_name": project_name,
            "project_id": project_id,
            "project_name_camel_case": "".join([each.capitalize() for each in project_name.split("_")]),
            "leaf_version": leaf_version,
            "created_at": created,
            "last_update": created,
        },
        no_input=True,
    )

    print(f"project [{name}] created.")
    raise typer.Exit()


@app.command(name="complete-project", help="using gpt-4 to roughly complete project code")
def complete_project(
    target: Annotated[str, typer.Option(metavar="target_dir")],
    reference: Annotated[Optional[str], typer.Option(metavar="reference_dir")] = None,
    api_key: Annotated[Optional[str], typer.Option(metavar="openai_api_key")] = None,
    disable_definition_completion: Annotated[bool, typer.Option()] = False,
    disable_agent_completion: Annotated[bool, typer.Option()] = False,
    disable_scene_completion: Annotated[bool, typer.Option()] = False,
):
    from .project_completion import Pipeline, PipelineConfig

    config = PipelineConfig(
        target_project=target,
        reference_project=reference,
        openai_api_key=api_key,
        definition_completion=not disable_definition_completion,
        agent_completion=not disable_agent_completion,
        scene_completion=not disable_scene_completion,
    )
    pipeline = Pipeline(config)
    pipeline.run()


@app.command(name="update-project-structure", help="sync project structure to the template.")
def update_project_structure(target: Annotated[str, typer.Argument(metavar="target_dir")]):
    target = os.path.abspath(target)
    dot_leaf_dir = os.path.join(target, ".leaf")
    if not os.path.exists(dot_leaf_dir):
        raise typer.BadParameter("not a leaf playground project.")
    with open(os.path.join(dot_leaf_dir, "project_config.json"), "r", encoding="utf-8") as f:
        project_config = json.load(f)
    project_name = project_config["name"]
    project_id = project_config.get("id", project_name + "_" + uuid4().hex[:8])
    created_at = project_config.get("created_at", "unknown")
    last_update = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    cookiecutter(
        output_dir=os.path.dirname(target),
        template=template_dir,
        extra_context={
            "project_name": project_name,
            "project_id": project_id,
            "project_name_camel_case": "".join([each.capitalize() for each in project_name.split("_")]),
            "leaf_version": leaf_version,
            "created_at": created_at,
            "last_update": last_update,
        },
        no_input=True,
        overwrite_if_exists=True,
        skip_if_file_exists=True,
        keep_project_on_failure=True,
    )

    # TODO: update project_config if there is update on field level

    publish_project(target, project_config["version"])

    print(f"project [{project_name}] updated.")
    raise typer.Exit()


def _replace_version_in_dockerfile(file_path, new_version):
    if not os.path.exists(file_path):
        return

    version_pattern = r"(leaf-playground==)(\d+\.\d+\.\d+[^\s]*)"

    with open(file_path, "r") as file:
        content = file.read()

    new_content = re.sub(version_pattern, r"\g<1>" + new_version, content)

    with open(file_path, "w") as file:
        file.write(new_content)


@app.command(
    name="publish",
    help="publish project, at this time, will only copy the newest app.py and update project_config.json",
)
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
    project_config["id"] = project_config.get("id", project_name + "_" + uuid4().hex[:8])
    project_config["version"] = version_str
    project_config["leaf_version"] = leaf_version
    project_config["created_at"] = project_config.get("created_at", "unknown")
    project_config["last_update"] = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")

    # get and set metadata
    pkg_root = Path(os.path.join(target, project_name))
    sys.path.insert(0, pkg_root.parent.as_posix())
    scene_class: Type[Scene] = relevantly_find_subclasses(
        root_path=pkg_root.as_posix(), prefix=project_name, base_class=Scene
    )[0]
    agent_classes: List[Type[SceneAgent]] = relevantly_find_subclasses(
        root_path=os.path.join(pkg_root, "agents"), prefix=f"{project_name}.agents", base_class=SceneAgent
    )
    evaluator_classes: List[Type[MetricEvaluator]] = relevantly_find_subclasses(
        root_path=os.path.join(pkg_root, "metric_evaluators"),
        prefix=f"{project_name}.metric_evaluators",
        base_class=MetricEvaluator,
    )
    chart_classes: List[Type[Chart]] = relevantly_find_subclasses(
        root_path=os.path.join(pkg_root, "charts"), prefix=f"{project_name}.charts", base_class=Chart
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
        "evaluators_metadata": (
            [
                evaluator_cls.get_metadata().model_dump(mode="json", by_alias=True)
                for evaluator_cls in evaluator_classes
            ]
            if evaluator_classes
            else None
        ),
        "charts_metadata": (
            [chart_cls.get_metadata().model_dump(mode="json", by_alias=True) for chart_cls in chart_classes]
            if chart_classes
            else None
        ),
    }

    with open(os.path.join(dot_leaf_dir, "project_config.json"), "w", encoding="utf-8") as f:
        json.dump(project_config, f, indent=4, ensure_ascii=False)

    app_py_file_path = os.path.join(template_dir, "{{cookiecutter.project_name}}", ".leaf", "app.py")
    shutil.copy(app_py_file_path, os.path.join(dot_leaf_dir, "app.py"))

    _replace_version_in_dockerfile(os.path.join(target, "Dockerfile"), leaf_version)

    print(f"publish new version [{version_str}]")
    raise typer.Exit()


__web_ui_release_site__ = (
    "https://github.com/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-webui/releases/download"
)
__web_ui_release_list_url__ = (
    "https://api.github.com/repos/LLM-Evaluation-s-Always-Fatiguing/leaf-playground-webui/releases?per_page=100"
)


def get_latest_webui_releases(cli_version_str: str):
    # cli_version like "0.4.0", try to get the latest web ui version starting with "v0.4."
    cli_version = version.parse(cli_version_str)
    webui_version_prefix = f"v{cli_version.major}.{cli_version.minor}."

    response = requests.get(__web_ui_release_list_url__)
    if response.status_code != 200:
        raise requests.RequestException(
            f"request {__web_ui_release_list_url__} to get releases failed with [{response.status_code}]: "
            f"{response.content}"
        )
    releases = response.json()
    for release in releases:
        if release["tag_name"].startswith(webui_version_prefix):
            return release["tag_name"]

    raise ValueError(f"can't find any web ui release version starts with {webui_version_prefix}")


def download_web_ui(cli_version: str) -> str:
    leaf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".leaf_workspace")
    tmp_dir = os.path.join(leaf_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    web_ui_dir = os.path.join(leaf_dir, "web_ui")
    os.makedirs(web_ui_dir, exist_ok=True)

    latest_web_ui_version = get_latest_webui_releases(cli_version)

    web_ui_download_url = __web_ui_release_site__ + f"/{latest_web_ui_version}/webui-{latest_web_ui_version}.zip"
    web_ui_hash_url = __web_ui_release_site__ + f"/{latest_web_ui_version}/webui-{latest_web_ui_version}.zip.sha256"

    web_ui_file_name = os.path.split(web_ui_download_url)[-1]
    web_ui_hash_file_name = os.path.split(web_ui_hash_url)[-1]

    web_ui_save_dir = os.path.join(web_ui_dir, os.path.splitext(web_ui_file_name)[0])
    web_ui_file_save_path = os.path.join(tmp_dir, web_ui_file_name)

    # check hash is the same
    remote_hash = requests.get(web_ui_hash_url).content.decode(encoding="utf-8")
    local_hash_save_path = os.path.join(web_ui_dir, web_ui_hash_file_name)
    if os.path.exists(local_hash_save_path):
        with open(local_hash_save_path, "r", encoding="utf-8") as f:
            local_hash = f.read().strip()
        if remote_hash == local_hash and os.path.exists(web_ui_save_dir):
            return web_ui_save_dir
    with open(local_hash_save_path, "w", encoding="utf-8") as f:
        f.write(remote_hash)

    # (re-)download web ui
    if os.path.exists(web_ui_save_dir):
        shutil.rmtree(web_ui_save_dir)
    try:
        response = urlopen(web_ui_download_url)
        size = int(response.headers["Content-Length"])
        with open(web_ui_file_save_path, "wb") as local_file:
            with wrap_file(response, size, description="download web ui...") as remote_file:
                for chunk in remote_file:
                    local_file.write(chunk)
    except:
        # download failed, clean
        print("download failed, clean caches")
        if os.path.exists(local_hash_save_path):
            os.remove(local_hash_save_path)
        if os.path.exists(web_ui_file_save_path):
            os.remove(web_ui_file_save_path)
        raise
    with zipfile.ZipFile(web_ui_file_save_path, "r") as zip_ref:
        zip_ref.extractall(web_ui_save_dir)
    os.remove(web_ui_file_save_path)

    return web_ui_save_dir


@app.command(name="start-server", help="start a leaf-playground server, will firstly download WEB UI if necessary.")
def start_server(
    hub_dir: Annotated[str, typer.Option()] = os.getcwd(),
    port: Annotated[int, typer.Option()] = 8000,
    ui_port: Annotated[int, typer.Option()] = 3000,
    dev_dir: Annotated[Optional[str], typer.Option()] = None,
    web_ui_dir: Annotated[Optional[str], typer.Option()] = None,
    no_web_ui: Annotated[Optional[bool], typer.Option()] = False,
    runtime_env: Annotated[str, typer.Option(click_type=click.Choice(["local", "docker"]))] = "local",
    db_type: Annotated[str, typer.Option(click_type=click.Choice(["sqlite", "postgresql"]))] = "sqlite",
    db_url: Annotated[Optional[str], typer.Option()] = None,
    debug: Annotated[bool, typer.Option()] = False,
    debugger_host: Annotated[str, typer.Option()] = "localhost",
    debugger_ide: Annotated[str, typer.Option(click_type=click.Choice(["pycharm", "vscode"]))] = "pycharm",
    debugger_port_for_server: Annotated[int, typer.Option()] = 3456,
    debugger_port_for_project: Annotated[int, typer.Option()] = 3457,
):
    if not no_web_ui:
        if web_ui_dir and not os.path.isdir(web_ui_dir):
            raise typer.BadParameter("value of argument '--web_ui' must be a local existed directory")
        if not web_ui_dir:
            web_ui_dir = download_web_ui(leaf_version)
        server_url = f"http://127.0.0.1:{port}"
        subprocess.Popen(f"node {os.path.join(web_ui_dir, 'start.js')} --port={ui_port} --server={server_url}".split())

    if dev_dir:
        dev_dir = os.path.abspath(dev_dir)
        sys.path.insert(0, dev_dir)

    from .server.app import config_server, AppConfig
    from .server.utils import get_local_ip
    from .server.task.db import DBType
    from .server.task.model import TaskRunTimeEnv
    from .utils.debug_utils import DebuggerConfig, IDEType

    if not os.path.exists(hub_dir):
        raise typer.BadParameter(f"zoo [{hub_dir}] not exist.")

    if debug and runtime_env != "local":
        reset_to_local = input(
            "You are using debug mode, which currently only support runtime_env='local', "
            f"but get runtime_env='{runtime_env}', reset to 'local'?(y/n):"
        )
        if reset_to_local.lower() == "y":
            runtime_env = "local"
        else:
            raise typer.BadParameter("can only use debug mode when runtime_env='local'")

    config_server(
        config=AppConfig(
            hub_dir=Path(hub_dir),
            server_port=port,
            server_host=get_local_ip(),
            runtime_env=TaskRunTimeEnv[runtime_env.upper()],
            db_type=DBType.SQLite if db_type == "sqlite" else DBType.PostgreSQL,
            db_url=db_url,
            server_debugger_config=DebuggerConfig(
                ide_type=IDEType.PyCharm if debugger_ide == "pycharm" else IDEType.VSCode,
                host=debugger_host,
                port=debugger_port_for_server,
                debug=debug
            ),
            project_debugger_config=DebuggerConfig(
                ide_type=IDEType.PyCharm if debugger_ide == "pycharm" else IDEType.VSCode,
                host=debugger_host,
                port=debugger_port_for_project,
                debug=debug
            ),
        )
    )

    import uvicorn

    uvicorn.run(
        "leaf_playground_cli.server.app:app", port=port, host="0.0.0.0", reload=bool(dev_dir), reload_dirs=dev_dir
    )


@app.command(name="version", help="currently installed leaf-playground framework version")
def get_version():
    print(f"v{leaf_version}")


@app.command(name="web-ui-version", help="bounded web ui version currently installed leaf-playground framework uses")
def get_web_ui_version():
    print(get_latest_webui_releases(leaf_version))


# TODO: add command to migrate database using Alembic


if __name__ == "__main__":
    app()
