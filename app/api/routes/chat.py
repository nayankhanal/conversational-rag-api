from fastapi import APIRouter, status

from app.api.deps import ChatServiceDep, ContainerDep
from app.schemas.chat import ChatMessage, ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    return await service.handle(request.session_id, request.message)


@router.get("/{session_id}/history", response_model=list[ChatMessage])
async def get_history(session_id: str, container: ContainerDep) -> list[ChatMessage]:
    return await container.memory.get_history(session_id)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(session_id: str, container: ContainerDep) -> None:
    await container.memory.clear(session_id)
