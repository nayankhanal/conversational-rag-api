from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI

from app.api.routes import chat, documents
from app.core.config import get_settings
from app.core.container import Container
from app.core.exceptions import register_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = await Container.create(get_settings())
    app.state.container = container
    try:
        yield
    finally:
        await container.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Backend", version="1.0.0", lifespan=lifespan)
    register_exception_handlers(app)

    api = APIRouter(prefix="/api/v1")
    api.include_router(documents.router)
    api.include_router(chat.router)
    app.include_router(api)
    return app


app = create_app()
