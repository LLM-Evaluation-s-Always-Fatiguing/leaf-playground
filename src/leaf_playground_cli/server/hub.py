import base64
import json
import os
import traceback
import warnings
from typing import Dict, List, Optional, Any

import aiohttp
from fastapi import status, APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, PrivateAttr

from leaf_playground._type import Singleton


hub_router = APIRouter(prefix="/hub")


class ProjectMetadata(BaseModel):
    scene_metadata: dict = Field(default=dict())
    agents_metadata: Dict[str, List[dict]] = Field(default=dict())
    evaluators_metadata: Optional[List[dict]] = Field(default=None)
    charts_metadata: Optional[List[dict]] = Field(default=None)


class Project(BaseModel):
    # fields load from project config
    name: str = Field(default=...)
    id: str = Field(default=...)
    version: str = Field(default=...)
    leaf_version: str = Field(default=...)
    created_at: str = Field(default=...)
    last_update: str = Field(default=...)
    metadata: ProjectMetadata = Field(default=...)

    # fields set in run time
    work_dir: str = Field(default=...)
    readme: str = Field(default="")
    requirements: str = Field(default="")
    dockerfile: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        self.readme = self._get_readme()
        self.requirements = self._get_requirements()
        self.dockerfile = self._get_dockerfile()

    @property
    def has_readme(self) -> bool:
        return bool(self.readme)

    @property
    def has_requirements(self) -> bool:
        return bool(self.requirements)

    @property
    def has_dockerfile(self) -> bool:
        return bool(self.dockerfile)

    @staticmethod
    def _read_file(fpath) -> str:
        if not os.path.exists(fpath):
            return ""
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def _get_readme(self) -> str:
        return self._read_file(os.path.join(self.work_dir, "README.md"))

    def _get_requirements(self) -> str:
        return self._read_file(os.path.join(self.work_dir, "requirements.txt"))

    def _get_dockerfile(self) -> str:
        return self._read_file(os.path.join(self.work_dir, "Dockerfile"))

    def update_readme(self):
        self.readme = self._get_readme()

    def update_requirements(self):
        self.requirements = self._get_requirements()

    def update_dockerfile(self):
        self.dockerfile = self._get_dockerfile()

    def update_config(self):
        with open(os.path.join(self.work_dir, ".leaf", "project_config.json"), "r", encoding="utf-8") as f:
            proj_config = json.load(f)
        for k, v in proj_config.items():
            if k in self.model_fields_set:
                if k != "metadata":
                    setattr(self, k, v)
                else:
                    setattr(self, k, ProjectMetadata(**v))

    def update(self):
        self.update_config()
        self.update_readme()
        self.update_requirements()
        self.update_dockerfile()


class Hub(Singleton):
    def __init__(self, hub_dir: str):
        self._hub_dir = hub_dir

        self._projects: Dict[str, Project] = {}
        self.scan_projects()

    @property
    def hub_dir(self) -> str:
        return self._hub_dir

    @property
    def projects(self) -> Dict[str, Project]:
        return self._projects

    def scan_projects(self) -> None:
        def _scan_projects(hub_dir: str) -> List[Project]:
            projects_list = []

            if not os.path.isdir(hub_dir) or not os.path.exists(hub_dir):
                return projects_list

            if os.path.exists(os.path.join(hub_dir, ".leaf")):
                work_dir = hub_dir
                with open(os.path.join(work_dir, ".leaf", "project_config.json"), "r", encoding="utf-8") as f:
                    try:
                        project = Project(**json.load(f), work_dir=work_dir)
                    except:
                        traceback.print_exc()
                        warnings.warn(
                            f"load project config from {os.path.join(work_dir, '.leaf')} failed, "
                            "will ignore this project."
                        )
                    else:
                        project.work_dir = work_dir
                        projects_list.append(project)
            else:
                for root, dirs, _ in os.walk(hub_dir):
                    for work_dir in dirs:
                        projects_list += _scan_projects(str(os.path.join(root, work_dir)))

            return projects_list

        self._projects = {proj.id: proj for proj in _scan_projects(self._hub_dir)}

    def get_project(self, project_id: str) -> Project:
        try:
            project = self._projects[project_id]
        except KeyError:
            raise KeyError(f"project [{project_id}] not found, make sure you have this project under {self._hub_dir}")
        return project

    @classmethod
    def http_get_project_by_id(cls, project_id: str) -> Project:
        try:
            hub = cls.get_instance()
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="service not ready.")

        try:
            project = hub.get_project(project_id)
        except KeyError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        return project


@hub_router.get("/refresh", response_class=JSONResponse)
async def refresh_projects(hub: Hub = Depends(Hub.get_instance)) -> JSONResponse:
    """refresh projects, this will re-scan projects under the hub dir, respond a list of updated project id"""

    hub.scan_projects()
    return JSONResponse(content=list(hub.projects.keys()))


@hub_router.get("/{project_id}/info", response_model=Project)
async def get_project_info(project: Project = Depends(Hub.http_get_project_by_id)) -> Project:
    """retrieve a project's information by giving its id, this will reload local files to update first."""
    project.update()
    return project


@hub_router.get("/{project_id}/readme.md", response_class=PlainTextResponse)
async def get_project_readme(project: Project = Depends(Hub.http_get_project_by_id)) -> PlainTextResponse:
    """retrieve a project's README.md by giving its id, this will reload local file to update first"""
    project.update_readme()
    return PlainTextResponse(content=project.readme)


@hub_router.get("/{project_id}/requirements.txt", response_class=PlainTextResponse)
async def get_project_requirements(project: Project = Depends(Hub.http_get_project_by_id)) -> PlainTextResponse:
    """retrieve a project's requirements.txt by giving its id, this will reload local file to update first"""
    project.update_requirements()
    return PlainTextResponse(content=project.requirements)


@hub_router.get("/{project_id}/dockerfile", response_class=PlainTextResponse)
async def get_project_dockerfile(project: Project = Depends(Hub.http_get_project_by_id)) -> PlainTextResponse:
    """retrieve a project's Dockerfile by giving its id, this will reload local file to update first"""
    project.update_dockerfile()
    return PlainTextResponse(content=project.dockerfile)


def validate_file_path(file_path: str):
    if file_path.startswith(".") and not file_path.startswith("./"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="relevant path must starts with './'"
        )
    return file_path


@hub_router.get("/{project_id}/assets/download_image", response_class=JSONResponse)
async def get_project_image_asset(
    project: Project = Depends(Hub.http_get_project_by_id), file_path: str = Depends(validate_file_path)
) -> JSONResponse:
    """
    retrieve a project's image asset by giving project_id and file_path, support both local path and http url,
    if given local path is a relevant path, it must start with './', otherwise responds with 422 status code
    """
    if file_path.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as session:
            async with session.get(file_path) as response:
                if response.status == 200:
                    image_data = await response.read()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, detail=f"can't connect to URL [{file_path}]."
                    )
    else:
        if file_path.startswith("."):
            file_path = os.path.join(project.work_dir, file_path[2:])
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"local file [{file_path}] not found.")
        with open(file_path, "rb") as f:
            image_data = f.read()

    return JSONResponse(content={"image": base64.b64encode(image_data).decode()})


# TODO: support download audio and video


__all__ = ["hub_router", "ProjectMetadata", "Project", "Hub"]
