from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status_code: int = 500

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedFileTypeError(AppError):
    status_code = 415


class FileTooLargeError(AppError):
    status_code = 413


class EmptyDocumentError(AppError):
    status_code = 422


class InvalidChunkingError(AppError):
    status_code = 422


class NotFoundError(AppError):
    status_code = 404


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
