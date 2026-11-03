import asyncio
from io import BytesIO
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.exceptions import EmptyDocumentError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _extract_pdf(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except PyPdfError as exc:
        raise EmptyDocumentError(f"Could not read PDF: {exc}") from exc


def _extract_txt(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


async def extract_text(filename: str, data: bytes) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError("Only .pdf and .txt files are supported.")
    extractor = _extract_pdf if extension == ".pdf" else _extract_txt
    text = (await asyncio.to_thread(extractor, data)).strip()
    if not text:
        raise EmptyDocumentError("No extractable text found in the document.")
    return text
