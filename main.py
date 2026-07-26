import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from core import get_purnank_response, get_purnank_response_stream

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    n_results: int = Field(default=3, ge=1, le=10)
    session_id: str | None = Field(default=None)
    use_ensemble: bool = Field(default=False)
    history: list[dict] | None = Field(
        default=None,
        description="Full conversation history to use for context. If provided, session_id is ignored."
    )

# Session storage remains for backward compatibility, but we'll prioritize history
_sessions: dict[str, list[dict]] = {}
_session_touched: dict[str, float] = {}
SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_SESSIONS = 1000

def _prune_sessions():
    now = time.time()
    expired = [sid for sid, ts in _session_touched.items() if now - ts > SESSION_TTL_SECONDS]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_touched.pop(sid, None)
    if len(_sessions) > MAX_SESSIONS:
        oldest = sorted(_session_touched.items(), key=lambda kv: kv[1])[: len(_sessions) - MAX_SESSIONS]
        for sid, _ in oldest:
            _sessions.pop(sid, None)
            _session_touched.pop(sid, None)

def _get_history(session_id):
    return _sessions.get(session_id, []) if session_id else []

def _append_turn(session_id, question, answer):
    if not session_id:
        return
    _sessions.setdefault(session_id, [])
    _sessions[session_id].append({"role": "user", "content": question})
    _sessions[session_id].append({"role": "assistant", "content": answer})
    _sessions[session_id] = _sessions[session_id][-16:]
    _session_touched[session_id] = time.time()
    _prune_sessions()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/session")
async def create_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = []
    _session_touched[sid] = time.time()
    return {"session_id": sid}

@app.post("/ask")
async def ask(q: Query):
    try:
        # Prefer client-provided history if available
        if q.history is not None:
            history = q.history
            session_id = q.session_id or str(uuid.uuid4())  # still generate for response
        else:
            session_id = q.session_id or str(uuid.uuid4())
            history = _get_history(session_id)

        answer, sources, suggestions = await run_in_threadpool(
            get_purnank_response, q.question, q.n_results, history, q.use_ensemble
        )

        # If we used server session, append turn
        if q.history is None:
            _append_turn(session_id, q.question, answer)

        return {
            "answer": answer,
            "sources": sources,
            "suggestions": suggestions,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/ask/stream")
async def ask_stream(q: Query):
    session_id = q.session_id or str(uuid.uuid4())
    history = _get_history(session_id)

    def token_generator():
        collected = []
        try:
            for chunk in get_purnank_response_stream(q.question, q.n_results, history):
                collected.append(chunk)
                yield chunk
        except Exception as e:
            yield f"\n\n⚠️ Something went wrong: {e}"
        finally:
            if collected:
                _append_turn(session_id, q.question, "".join(collected))

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(
        content=content,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

app.mount("/static", StaticFiles(directory="static"), name="static_files")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)