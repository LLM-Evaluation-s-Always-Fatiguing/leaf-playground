import base64
import json
import os
import traceback
import warnings
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union
from typing_extensions import Annotated

import aiohttp
from fastapi import status, APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, PrivateAttr

from ._type import SingletonClass


hub_router = APIRouter(prefix="/hub")


class ProjectMetadata(BaseModel):
    scene_metadata: dict = Field(default=dict())
    agents_metadata: Dict[str, List[dict]] = Field(default=dict())
    evaluators_metadata: Optional[List[dict]] = Field(default=None)
    charts_metadata: Optional[List[dict]] = Field(default=None)


class Project(BaseModel):
    name: str = Field(default=...)
    id: str = Field(default=...)
    version: str = Field(default=...)
    leaf_version: str = Field(default=...)
    created_at: str = Field(default=...)
    last_update: str = Field(default=...)
    metadata: ProjectMetadata = Field(default=...)

    _work_dir: str = PrivateAttr(default="")

    @property
    def work_dir(self):
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: str):
        self._work_dir = work_dir

    @property
    def readme_fpath(self) -> str:
        return os.path.join(self.work_dir, "README.md")

    @property
    def requirements_fpath(self) -> str:
        return os.path.join(self.work_dir, "requirements.txt")

    @property
    def dockerfile_fpath(self) -> str:
        return os.path.join(self.work_dir, "Dockerfile")

    @property
    def has_readme(self) -> bool:
        return os.path.exists(self.readme_fpath)

    @property
    def has_requirements(self) -> bool:
        return os.path.exists(self.requirements_fpath)

    @property
    def has_dockerfile(self) -> bool:
        return os.path.exists(self.dockerfile_fpath)

    @staticmethod
    def _read_file(fpath) -> str:
        if not os.path.exists(fpath):
            return ""
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        return content

    def get_readme(self) -> str:
        return self._read_file(self.readme_fpath)

    def get_requirements(self) -> str:
        return self._read_file(self.requirements_fpath)

    def get_dockerfile(self) -> str:
        return self._read_file(self.dockerfile_fpath)


class Hub(SingletonClass):
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
                with open(
                    os.path.join(work_dir, ".leaf", "project_config.json"), "r", encoding="utf-8"
                ) as f:
                    try:
                        project = Project(**json.load(f))
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


@hub_router.get("/{project_id}/readme.md", response_class=PlainTextResponse)
async def get_project_readme(project: Project = Depends(Hub.http_get_project_by_id)) -> PlainTextResponse:
    if not project.has_readme:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"README.md for project [{project.id}] not found."
        )
    return PlainTextResponse(content=project.get_readme())


@hub_router.get("/{project_id}/requirements.txt", response_class=PlainTextResponse)
async def get_project_requirements(project: Project = Depends(Hub.http_get_project_by_id)) -> PlainTextResponse:
    if not project.has_requirements:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"requirements.txt for project [{project.id}] not found."
        )
    return PlainTextResponse(content=project.get_requirements())


@hub_router.get("/{project_id}/dockerfile", response_class=PlainTextResponse)
async def get_project_dockerfile(project: Project = Depends(Hub.http_get_project_by_id)) -> PlainTextResponse:
    if not project.has_dockerfile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Dockerfile for project [{project.id}] not found."
        )
    return PlainTextResponse(content=project.get_dockerfile())


def validate_file_path(file_path: str):
    if file_path.startswith(".") and not file_path.startswith("./"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="relevant path must starts with './'"
        )
    return file_path


@hub_router.get("/{project_id}/assets/download_image", response_class=JSONResponse)
async def get_image(
    project: Project = Depends(Hub.http_get_project_by_id),
    file_path: str = Depends(validate_file_path),
) -> JSONResponse:
    if file_path.startswith(("http://", "https://")):
        async with aiohttp.ClientSession() as session:
            async with session.get(file_path) as response:
                if response.status == 200:
                    image_data = await response.read()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"can't connect to URL [{file_path}]."
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
