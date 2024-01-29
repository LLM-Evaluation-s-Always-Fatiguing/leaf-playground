import asyncio
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Callable, List, Optional

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import select, Select

from leaf_playground._type import Singleton
from leaf_playground.core.scene_engine import SceneEngineState
from leaf_playground.utils.thread_util import run_asynchronously

from .model import *


class DBType(Enum):
    SQLite = "SQLite"
    PostgreSQL = "PostgreSQL"


class DB(Singleton):
    def __init__(self, db_type: DBType, db_url: str):
        if db_type == DBType.SQLite:
            self.url = f"sqlite+aiosqlite:///{db_url}"
        else:
            self.url = f"postgresql+asyncpg://{db_url}"
        self.engine = create_async_engine(self.url)
        asyncio.ensure_future(self._on_startup())
        asyncio.ensure_future(self._background_job())

        self._run = asyncio.Event()

    async def wait_db_startup(self):
        await self._run.wait()

    async def _on_startup(self):
        async with self.engine.begin() as conn:
            # 创建所有表结构
            await conn.run_sync(SQLModel.metadata.create_all)
        self._run.set()

    async def _background_job(self):
        await self._run.wait()

        deleting_tasks = await self.get_task_by_life_cycle(TaskDBLifeCycle.DELETING)
        for task in deleting_tasks:
            await self.delete_task(task.id)

    async def _insert(self, data: SQLModel) -> int:
        async with AsyncSession(self.engine) as session:
            try:
                session.add(data)
                await session.commit()
                succeed = True
            except:
                succeed = False
        return 200 if succeed else 500

    async def _update(self, obj_getter: Callable[[AsyncSession], Any], **kwargs) -> int:
        async with AsyncSession(self.engine) as session:
            obj = await run_asynchronously(obj_getter, session)
            if not obj:
                return 404
            for k, v in kwargs.items():
                try:
                    setattr(obj, k, v)
                except:
                    return 422
            try:
                session.add(obj)
                await session.commit()
            except:
                return 500
        return 200

    async def insert_task(self, task: Task):
        status_code = await self._insert(TaskTable.from_task(task))
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=f"insert task [{task.id}] failed.")

    async def update_task_status(self, tid: str, status: str):
        async def obj_getter(session: AsyncSession):
            return await session.get(TaskTable, tid)

        valid_status = [s.value for s in SceneEngineState]
        if status not in valid_status:
            raise HTTPException(status_code=422, detail=f"{status} is not valid, valid status are {valid_status}")

        status_code = await self._update(obj_getter, **{"status": status})
        if status_code == 404:
            raise HTTPException(status_code=status_code, detail=f"task [{tid}] not found")
        if status_code == 422:
            raise HTTPException(status_code=status_code, detail=f"task [{tid}] update failed")
        if status_code == 500:
            raise HTTPException(
                status_code=500,
                detail=f"task [{tid}] update succeeded but session commit failed, maybe try again later",
            )

    async def get_task(self, tid: str) -> Task:
        async with AsyncSession(self.engine) as session:
            task = await session.get(TaskTable, tid)
            if not task:
                raise HTTPException(status_code=404, detail=f"task [{tid}] not found")
        return task.to_task()

    async def get_task_by_life_cycle(self, life_cycle: TaskDBLifeCycle) -> List[Task]:
        async with AsyncSession(self.engine) as session:
            tasks = (
                (await session.execute(select(TaskTable).where(TaskTable.live_cycle == life_cycle))).scalars().all()
            )
        if not tasks:
            return []
        return [task.to_task() for task in tasks]

    async def delete_task(self, tid: str):
        async def _delete_logs_batchify(session: AsyncSession):
            while True:
                try:
                    result = await session.execute(select(LogTable).where(LogTable.tid == tid).limit(100))
                    logs = result.scalars().all()
                    if not logs:
                        break
                    for log in logs:
                        await session.delete(log)
                    await session.commit()
                except:
                    await session.rollback()
                    raise

        async def _delete_messages_batchify(session: AsyncSession):
            while True:
                try:
                    result = await session.execute(select(MessageTable).where(MessageTable.tid == tid).limit(100))
                    messages = result.scalars().all()
                    if not messages:
                        break
                    for message in messages:
                        await session.delete(message)
                    await session.commit()
                except:
                    await session.rollback()
                    raise

        async def _delete_task():
            session = AsyncSession(self.engine)
            task = await session.get(TaskTable, tid)
            if not task:
                await session.close()
                return

            # update task live cycle to "deleting"
            try:
                if task.live_cycle != TaskDBLifeCycle.DELETING:
                    task.live_cycle = TaskDBLifeCycle.DELETING
                    await session.commit()
            except:
                traceback.print_exc()
                await session.rollback()
                await session.close()
                raise
            try:
                # delete logs first
                await _delete_logs_batchify(session)
                # then delete messages
                await _delete_messages_batchify(session)
            except:
                traceback.print_exc()
                await session.close()
                raise
            # finally delete the task itself
            try:
                await session.delete(task)
                await session.commit()
            except:
                traceback.print_exc()
                await session.rollback()
                await session.close()
                raise

        await _delete_task()  # TODO: using aio-pika to do delete in truly background

    async def insert_log(self, log: Log):
        status_code = await self._insert(LogTable.from_log(log))
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=f"insert log [{log.id}] failed")

    async def update_log(self, new_log: Log):
        log_id = new_log.id

        async def obj_getter(session: AsyncSession):
            return await session.get(LogTable, log_id)

        status_code = await self._update(obj_getter, **new_log.model_dump())
        if status_code == 404:
            raise HTTPException(status_code=status_code, detail=f"log [{log_id}] not found")
        if status_code == 422:
            raise HTTPException(status_code=status_code, detail=f"log [{log_id}] update failed")
        if status_code == 500:
            raise HTTPException(
                status_code=500,
                detail=f"log [{log_id}] update succeeded but session commit failed, maybe try again later",
            )

    async def get_log_by_id(self, log_id: str) -> Log:
        async with AsyncSession(self.engine) as session:
            log = await session.get(LogTable, log_id)
            if not log:
                raise HTTPException(status_code=404, detail=f"log [{log_id}] not found")
        return log.to_log()

    async def get_logs_by_tid_with_update_time_constraint(
        self, tid: str, last_checked_dt: Optional[datetime] = None
    ) -> List[Log]:
        if not last_checked_dt:
            statement: Select = select(LogTable).where(LogTable.tid == tid)
        else:
            statement: Select = select(LogTable).where(LogTable.tid == tid, LogTable.db_last_update > last_checked_dt)
        statement = statement.order_by(LogTable.db_last_update if last_checked_dt is not None else LogTable.created_at)
        async with AsyncSession(self.engine) as session:
            logs = (await session.execute(statement)).scalars().all()
        if not logs:
            return []
        return [log.to_log() for log in logs]

    async def get_logs_by_tid_paginate(self, tid: str, skip: int = 0, limit: int = 20) -> List[Log]:
        async with AsyncSession(self.engine) as session:
            logs = (
                await session.execute(
                    select(LogTable).where(LogTable.tid == tid).order_by(LogTable.created_at).offset(skip).limit(limit)
                )
            ).scalars().all()
        if not logs:
            return []
        return [log.to_log() for log in logs]

    async def insert_message(self, message: Message):
        status_code = await self._insert(MessageTable.from_message(message))
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=f"insert message [{message.id}] failed")

    async def get_message_by_id(self, message_id: str) -> Message:
        async with AsyncSession(self.engine) as session:
            message = await session.get(MessageTable, message_id)
            if not message:
                raise HTTPException(status_code=404, detail=f"message [{message_id}] not found")
        return message.to_message()

    async def save_task_results(self, task_results: TaskResults):
        async def obj_getter(session: AsyncSession):
            return await session.get(TaskResultsTable, task_results.id)

        async with AsyncSession(self.engine) as session:
            task_results_exists = bool(await obj_getter(session))

        if not task_results_exists:
            status_code = await self._insert(TaskResultsTable.from_task_results(task_results))
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=f"insert task_results [{task_results.id}] failed")
        else:
            status_code = await self._update(obj_getter, **task_results.model_dump())
            if status_code == 422:
                raise HTTPException(status_code=status_code, detail=f"task_results [{task_results.id}] update failed")
            if status_code == 500:
                raise HTTPException(
                    status_code=500,
                    detail=f"task_results [{task_results.id}] update succeeded but session commit failed, "
                           f"maybe try again later",
                )

    async def get_task_results_by_id(self, tid: str):
        async with AsyncSession(self.engine) as session:
            task_results = await session.get(TaskResultsTable, tid)
            if not task_results:
                raise HTTPException(status_code=404, detail=f"task_results [{tid}] not found")
        return task_results.to_task_results()


__all__ = ["DBType", "DB"]
