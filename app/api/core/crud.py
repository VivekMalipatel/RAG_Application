from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union, Callable
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.db.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)

class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model
        self._process_create = None
        self._process_update = None

    def set_create_handler(self, handler: Callable):
        self._process_create = handler
        return self

    def set_update_handler(self, handler: Callable):
        self._process_update = handler
        return self

    async def get(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        async with db.begin():
            result = await db.execute(self.model.__table__.select().where(self.model.id == id))
            return result.scalars().first()

    async def get_multi(self, db: AsyncSession, *, skip: int = 0, limit: int = 100) -> List[ModelType]:
        async with db.begin():
            result = await db.execute(self.model.__table__.select().offset(skip).limit(limit))
            return result.scalars().all()

    async def create(self, db: AsyncSession, *, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = jsonable_encoder(obj_in)
        if self._process_create:
            obj_in_data = self._process_create(obj_in_data)

        db_obj = self.model(**obj_in_data)
        async with db.begin():
            db.add(db_obj)
        await db.refresh(db_obj)
        return db_obj

    async def update(self, db: AsyncSession, *, db_obj: ModelType, obj_in: Union[UpdateSchemaType, Dict[str, Any]]) -> ModelType:
        obj_data = jsonable_encoder(db_obj)
        update_data = obj_in if isinstance(obj_in, dict) else obj_in.dict(exclude_unset=True)

        if self._process_update:
            update_data = self._process_update(update_data)

        for field in obj_data:
            if field in update_data:
                setattr(db_obj, field, update_data[field])

        async with db.begin():
            db.add(db_obj)
        await db.refresh(db_obj)
        return db_obj

    async def delete(self, db: AsyncSession, *, id: int) -> ModelType:
        async with db.begin():
            obj = await db.get(self.model, id)
            await db.delete(obj)
        return obj