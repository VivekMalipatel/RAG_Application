from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ...db.base import get_db
from ...models.agent import Agent
from ...schemas.agent import AgentCreate, AgentUpdate, AgentResponse, DeleteAgentResponse
from ...core.crud import CRUDBase

router = APIRouter()

agent_crud = CRUDBase[Agent, AgentCreate, AgentUpdate](Agent)

@router.post("/", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(agent: AgentCreate, db: Session = Depends(get_db)):
    # Check if agent with same name exists
    existing_agent = db.query(Agent).filter(Agent.name == agent.name).first()
    if existing_agent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent with this name already exists"
        )
    
    return agent_crud.create(db=db, obj_in=agent)

@router.get("/", response_model=List[AgentResponse])
def read_agents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return agent_crud.get_multi(db=db, skip=skip, limit=limit)

@router.get("/{agent_id}", response_model=AgentResponse)
def read_agent(agent_id: int, db: Session = Depends(get_db)):
    agent = agent_crud.get(db=db, id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent

@router.put("/{agent_id}", response_model=AgentResponse)
def update_agent(
    agent_id: int,
    agent_update: AgentUpdate,
    db: Session = Depends(get_db)
):
    current_agent = agent_crud.get(db=db, id=agent_id)
    if not current_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    # Check if name is being changed and if it's already taken
    if agent_update.name != current_agent.name:
        existing_agent = db.query(Agent).filter(Agent.name == agent_update.name).first()
        if existing_agent:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent name already taken"
            )
    
    return agent_crud.update(db=db, db_obj=current_agent, obj_in=agent_update)

@router.delete("/{agent_id}", response_model=DeleteAgentResponse)
def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    agent = agent_crud.get(db=db, id=agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    name = agent.name
    agent_crud.delete(db=db, id=agent_id)
    
    return DeleteAgentResponse(
        message="Agent deleted successfully",
        name=name
    )