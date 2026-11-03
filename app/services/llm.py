import json
from collections.abc import Sequence
from datetime import date

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from app.schemas.chat import BookingDraft, ChatMessage
from app.services.protocols import RetrievedChunk, TurnAnalysis

_ANALYSIS_PROMPT = """\
You analyze one user turn in a conversation with a document-QA assistant that can also book \
interviews. Today is {today}.

Reply with ONLY a JSON object (no prose, no code fences) with exactly these keys:
- "wants_booking": true if the user wants to book/schedule an interview, OR is supplying details \
for a booking already in progress (current draft: {draft}); otherwise false.
- "standalone_question": the user's message rewritten so it is understandable without the \
conversation history (resolve pronouns and references). Copy it unchanged if already standalone.
- "name", "email", "date", "time": booking details found in THIS message or earlier user messages. \
"date" must be YYYY-MM-DD (resolve relative dates like "tomorrow" using today's date) and "time" \
must be 24-hour HH:MM. Use null for anything not provided. Never invent values."""

_ANSWER_PROMPT = """\
Answer the user's question using ONLY the context below. If the context does not contain the \
answer, say you don't know based on the uploaded documents. Be concise.

Context:
{context}"""


class _AnalysisPayload(BaseModel):
    wants_booking: bool = False
    standalone_question: str = ""
    name: str | None = None
    email: str | None = None
    date: str | None = None
    time: str | None = None


def _parse_payload(raw: str) -> _AnalysisPayload:
    """Providers differ on JSON mode, so extract the first {...} block leniently."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return _AnalysisPayload()
    try:
        return _AnalysisPayload.model_validate(json.loads(raw[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return _AnalysisPayload()


class OpenAIChatLLM:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def analyze_turn(
        self, history: Sequence[ChatMessage], message: str, draft: BookingDraft
    ) -> TurnAnalysis:
        system = _ANALYSIS_PROMPT.format(
            today=date.today().isoformat(), draft=draft.model_dump_json()
        )
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            *self._history_params(history),
            {"role": "user", "content": message},
        ]
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0,
        )
        payload = _parse_payload(completion.choices[0].message.content or "")
        return TurnAnalysis(
            wants_booking=payload.wants_booking,
            standalone_question=payload.standalone_question or message,
            booking=BookingDraft(
                name=payload.name, email=payload.email, date=payload.date, time=payload.time
            ),
        )

    async def answer(
        self, history: Sequence[ChatMessage], question: str, context: Sequence[RetrievedChunk]
    ) -> str:
        rendered = "\n\n".join(f"[{c.filename} #{c.chunk_index}] {c.text}" for c in context)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": _ANSWER_PROMPT.format(context=rendered or "(none)")},
            *self._history_params(history),
            {"role": "user", "content": question},
        ]
        completion = await self._client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.2
        )
        return completion.choices[0].message.content or ""

    @staticmethod
    def _history_params(history: Sequence[ChatMessage]) -> list[ChatCompletionMessageParam]:
        params: list[ChatCompletionMessageParam] = []
        for m in history:
            if m.role == "assistant":
                params.append({"role": "assistant", "content": m.content})
            else:
                params.append({"role": "user", "content": m.content})
        return params
