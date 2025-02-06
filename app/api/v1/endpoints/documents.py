from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ...db.base import get_db
from ...models.documents import Document
from ...schemas.documents import DocumentCreate, DocumentUpdate, DocumentResponse
from ...core.crud import CRUDBase

router = APIRouter()
document_crud = CRUDBase[Document, DocumentCreate, DocumentUpdate](Document)

@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(document: DocumentCreate, db: Session = Depends(get_db)):
    return document_crud.create(db=db, obj_in=document)

@router.get("/", response_model=List[DocumentResponse])
def read_documents(
    skip: int = 0, 
    limit: int = 100, 
    user_id: int | None = None,
    db: Session = Depends(get_db)
):
    query = db.query(Document)
    if user_id:
        query = query.filter(Document.user_id == user_id)
    return query.offset(skip).limit(limit).all()

@router.get("/{document_id}", response_model=DocumentResponse)
def read_document(document_id: int, db: Session = Depends(get_db)):
    document = document_crud.get(db=db, id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document

@router.put("/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: int,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db)
):
    current_document = document_crud.get(db=db, id=document_id)
    if not current_document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document_crud.update(
        db=db, 
        db_obj=current_document,
        obj_in=document_update
    )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = document_crud.get(db=db, id=document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    document_crud.delete(db=db, id=document_id)
    return None