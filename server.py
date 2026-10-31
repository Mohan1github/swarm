from __future__ import annotations

import base64
import binascii
import logging
from io import BytesIO
import json
import re
from pathlib import PurePath

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from auth import (
    AuthUser,
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    revoke_user_tokens,
    validate_email,
)
from agents.assistant_conversation_agent import (
    AssistantMemoryAccumulator,
    ConversationalAssistantAgent,
)
from agents.assistant_context_agent import AssistantContextAgent
from agents.proactive_conversation_agent import ProactiveConversationAgent
from agents.agent_output_vector_agent import AgentOutputVectorAgent
from agents.evidence_ledger_agent import (
    CitationAndTruthFinderAgent,
    EvidenceLedgerStore,
)
from agents.web_data_retriever_agent import PersistentWebVectorGraph
from activity_store import UserActivityStore
from agents.orchestrator import ReasoningSwarmEngine, WorkflowExecutionError
from graph_visualizer import render_live_execution_graph
from llm_client import build_llm_adapter, describe_provider_error
from vector_cache import VectorAnswerCache, VectorCacheHit

ANSWER_CACHE = VectorAnswerCache()
ACTIVITY_STORE = UserActivityStore()
ASSISTANT_MEMORY = AssistantMemoryAccumulator()
EVIDENCE_LEDGER_STORE = EvidenceLedgerStore()
LOGGER = logging.getLogger(__name__)
LIVE_WEB_QUERY_HINT = re.compile(
    r"\b(current|currently|today|tonight|tomorrow|this week|this month|this year|"
    r"latest|recent|recently|breaking|news|as of|up[- ]to[- ]date|updated|"
    r"real[- ]time|research|investigate|investigation|fact[- ]check|verify|verification|"
    r"credible sources?|citations?|stock|share price|market data|exchange rate|"
    r"financial market|interest rate|election|polls|weather|"
    r"information integrity|integrity check|sensitive|high[- ]stakes|"
    r"medical|health advice|legal advice|financial advice|election results?|"
    r"recall notice|active conflict|outbreak|regulation update)\b",
    re.IGNORECASE,
)


def _bypass_answer_cache(query: str, attachments: bool = False) -> bool:
    return attachments or bool(LIVE_WEB_QUERY_HINT.search(query))


class AttachmentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    media_type: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., max_length=7_000_000)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    total_agents: int = Field(default=10_000, ge=1, le=100_000)
    attachments: list[AttachmentRequest] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_attachment_payload_size(self):
        if sum(len(attachment.content) for attachment in self.attachments) > 15_000_000:
            raise ValueError("Combined attachment size must be 15 MB or less.")
        return self


class AssistantMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=3000)
    attachments: list[AttachmentRequest] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def validate_attachment_payload_size(self):
        if sum(len(attachment.content) for attachment in self.attachments) > 15_000_000:
            raise ValueError("Combined attachment size must be 15 MB or less.")
        return self


class QueryResponse(BaseModel):
    status: str
    mode: str
    query: str
    answer: str
    top_action: str
    action_probabilities: dict[str, float]
    validation_score: float
    validation_issues: list[str]
    selected_agents: int
    workflow_mode: str
    execution_graph: str
    metrics: dict
    execution_steps: list[dict]
    latency_ms: float
    model: str
    evidence_ledger: dict | None = None
    evidence_ledger_error: str | None = None


class CredentialsRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email_address(cls, value: str) -> str:
        if not validate_email(value):
            raise ValueError("Enter a valid email address.")
        return value.strip().lower()


class RegisterRequest(CredentialsRequest):
    password: str = Field(..., min_length=10, max_length=128)


class UserResponse(BaseModel):
    id: int
    email: str
    role: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


def _query_response(
    query: str,
    workflow: dict,
    cache_hit: VectorCacheHit | None = None,
) -> QueryResponse:
    actions = workflow["action_probabilities"]
    top_action = max(actions.items(), key=lambda item: item[1])[0]
    selected_agents = workflow["selected_agents"]
    validation_score = workflow["validation_score"]
    latency_ms = 0.0 if cache_hit else workflow["latency_ms"]
    execution_steps = workflow["steps"]
    execution_graph = render_live_execution_graph(
        actions,
        query,
        selected_agents=selected_agents,
        validation_score=validation_score,
        latency_ms=latency_ms,
        steps=execution_steps,
        route=workflow.get("route"),
    )
    metrics = {
        "validation_score": validation_score,
        "validation_issues": workflow["validation_issues"],
        "selected_agents": selected_agents,
        "top_action": top_action,
        "mode": "vector_cache" if cache_hit else "llm_live",
        "model": workflow["model"],
        "latency_ms": latency_ms,
        "completed_stages": len(execution_steps),
        "cache_hit": cache_hit is not None,
        "routing": workflow.get("route"),
    }
    if cache_hit:
        metrics["cache_similarity"] = round(cache_hit.similarity, 4)

    return QueryResponse(
        status="success",
        mode="vector_cache" if cache_hit else "llm_live",
        query=query,
        answer=workflow["answer"],
        top_action=top_action,
        action_probabilities=actions,
        validation_score=validation_score,
        validation_issues=workflow["validation_issues"],
        selected_agents=selected_agents,
        workflow_mode=(
            "vector_cache"
            if cache_hit
            else "direct_llm"
            if workflow.get("route", {}).get("decision") == "direct"
            else "sequential_llm_stages"
        ),
        execution_graph=execution_graph,
        metrics=metrics,
        execution_steps=execution_steps,
        latency_ms=latency_ms,
        model=workflow["model"],
        evidence_ledger=workflow.get("evidence_ledger"),
        evidence_ledger_error=workflow.get("evidence_ledger_error"),
    )


def _attach_evidence_ledger(workflow: dict, user_id: int, query: str, llm) -> None:
    sources = workflow.pop("retrieved_sources", [])
    if not (workflow.get("route") or {}).get("requires_web_search"):
        return
    try:
        workflow["evidence_ledger"] = CitationAndTruthFinderAgent(
            llm,
            store=EVIDENCE_LEDGER_STORE,
        ).analyze(
            user_id=user_id,
            query=query,
            answer=workflow.get("answer", ""),
            sources=sources,
        )
    except Exception as error:
        LOGGER.exception("Evidence ledger generation failed for user %s", user_id)
        workflow["evidence_ledger_error"] = (
            "The answer is ready, but its evidence ledger could not be completed. "
            f"({type(error).__name__})"
        )


app = FastAPI(
    title="Swarm Production API",
    version="1.0.0",
    description="FastAPI service for sequential, model-backed swarm workflow execution.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    engine = ReasoningSwarmEngine(total_agents=0)
    return {
        "status": "ok",
        "service": "swarm-production-api",
        "llm_configured": engine.llm.enabled,
        "model": engine.llm.model,
    }


def _auth_response(user: AuthUser) -> AuthResponse:
    return AuthResponse(
        access_token=create_access_token(user),
        token_type="bearer",
        user=user.public_data(),
    )


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> AuthResponse:
    user = create_user(payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists.")
    return _auth_response(user)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: CredentialsRequest) -> AuthResponse:
    user = authenticate_user(payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password is incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _auth_response(user)


@app.get("/api/auth/me", response_model=UserResponse)
def read_current_user(current_user: AuthUser = Depends(get_current_user)) -> UserResponse:
    return UserResponse(**current_user.public_data())


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(current_user: AuthUser = Depends(get_current_user)) -> Response:
    revoke_user_tokens(current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/activity/graph")
def read_activity_graph(
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    return ACTIVITY_STORE.graph(current_user.id)


@app.get("/api/evidence-ledger")
def read_evidence_ledger(
    limit: int = Query(default=20, ge=1, le=100),
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    return {"reports": EVIDENCE_LEDGER_STORE.list_reports(current_user.id, limit)}


@app.get("/api/assistant/history")
def read_assistant_history(
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    return {"turns": ASSISTANT_MEMORY.load_history(current_user.id)}


@app.delete("/api/history")
def clear_user_history(
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    try:
        ACTIVITY_STORE.clear_user(current_user.id)
        ASSISTANT_MEMORY.clear(current_user.id)
        ANSWER_CACHE.clear_user(current_user.id)
        AgentOutputVectorAgent().clear_user(current_user.id)
        PersistentWebVectorGraph().clear_user(current_user.id)
        EVIDENCE_LEDGER_STORE.clear_user(current_user.id)
    except Exception as error:
        LOGGER.exception("Could not clear all history for user %s", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not clear all history. Please retry.",
        ) from error
    return {"cleared": True}


@app.get("/api/assistant/session")
def read_assistant_session(
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    llm = build_llm_adapter()
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured.")
    turns, turn_count = ASSISTANT_MEMORY.load(current_user.id)
    memory_context = ASSISTANT_MEMORY.build_context(turns)
    activity_context = ""
    try:
        activity_context = AssistantContextAgent(llm).find_relevant_context(
            ACTIVITY_STORE.graph(current_user.id)
        )
    except Exception:
        LOGGER.exception("Could not load assistant activity context for user %s", current_user.id)
        activity_context = ""
    ASSISTANT_MEMORY.store_workspace_context(current_user.id, activity_context)
    try:
        opening = ProactiveConversationAgent(llm).opening(memory_context, activity_context)
    except Exception as error:
        LOGGER.exception("Assistant session opening failed for user %s", current_user.id)
        raise HTTPException(
            status_code=502,
            detail=f"The assistant could not start a conversation. {describe_provider_error(error)}",
        ) from error
    return {"messages": turns, "opening": opening, "memory_turn_count": turn_count}


def _assistant_sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=True)}\n\n"


@app.post("/api/assistant/session/stream")
def stream_assistant_session(
    current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    llm = build_llm_adapter()
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured.")

    def event_stream():
        try:
            turns, turn_count = ASSISTANT_MEMORY.load(current_user.id)
            yield _assistant_sse({
                "type": "session",
                "messages": turns,
                "memory_turn_count": turn_count,
            })
            memory_context = ASSISTANT_MEMORY.build_context(turns)
            activity_context = AssistantContextAgent(llm).find_relevant_context(
                ACTIVITY_STORE.graph(current_user.id)
            )
            ASSISTANT_MEMORY.store_workspace_context(current_user.id, activity_context)
            response = ""
            for chunk in ProactiveConversationAgent(llm).stream_opening(
                memory_context,
                activity_context,
            ):
                if not isinstance(chunk, str) or not chunk:
                    continue
                response += chunk
                yield _assistant_sse({"type": "delta", "text": chunk})
            if not response.strip():
                raise RuntimeError("The LLM provider returned an empty opening message.")
            turn_count = ASSISTANT_MEMORY.remember_assistant_turn(current_user.id, response)
            yield _assistant_sse({"type": "complete", "memory_turn_count": turn_count})
        except Exception as error:
            LOGGER.exception("Assistant proactive opening failed for user %s", current_user.id)
            yield _assistant_sse({
                "type": "error",
                "message": f"The assistant could not start a conversation. {describe_provider_error(error)}",
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/assistant/proactive/stream")
def stream_assistant_proactive_turn(
    current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    llm = build_llm_adapter()
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured.")

    def event_stream():
        try:
            turns, _ = ASSISTANT_MEMORY.load(current_user.id)
            memory_context = ASSISTANT_MEMORY.build_context(turns)
            activity_context = ASSISTANT_MEMORY.load_workspace_context(current_user.id)
            response = ""
            for chunk in ProactiveConversationAgent(llm).stream_follow_up(
                memory_context,
                activity_context,
            ):
                if not isinstance(chunk, str) or not chunk:
                    continue
                response += chunk
                yield _assistant_sse({"type": "delta", "text": chunk})
            if not response.strip():
                raise RuntimeError("The LLM provider returned an empty proactive message.")
            turn_count = ASSISTANT_MEMORY.remember_assistant_turn(current_user.id, response)
            yield _assistant_sse({
                "type": "complete",
                "memory_turn_count": turn_count,
            })
        except Exception as error:
            LOGGER.exception("Assistant proactive turn failed for user %s", current_user.id)
            yield _assistant_sse({
                "type": "error",
                "message": f"The assistant could not generate a proactive message. {describe_provider_error(error)}",
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/assistant/turn")
def send_assistant_turn(
    payload: AssistantMessageRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    llm = build_llm_adapter()
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured.")
    attachment_context, image_data_urls = _prepare_attachments(payload.attachments)
    turns, _ = ASSISTANT_MEMORY.load(current_user.id)
    memory_context = ASSISTANT_MEMORY.build_context(turns)
    try:
        answer = ConversationalAssistantAgent(llm).respond(
            payload.message,
            memory_context,
            attachment_context=attachment_context,
            image_data_urls=image_data_urls,
            activity_context=ASSISTANT_MEMORY.load_workspace_context(current_user.id),
        )
    except Exception as error:
        LOGGER.exception("Assistant response failed for user %s", current_user.id)
        raise HTTPException(
            status_code=502,
            detail=f"The assistant could not complete that reply. {describe_provider_error(error)}",
        ) from error
    memory_turn_count = ASSISTANT_MEMORY.remember_exchange(
        current_user.id,
        payload.message,
        answer,
    )
    return {"answer": answer, "memory_turn_count": memory_turn_count}


@app.post("/api/assistant/turn/stream")
def stream_assistant_turn(
    payload: AssistantMessageRequest,
    current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    llm = build_llm_adapter()
    if not llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured.")

    def event_stream():
        try:
            attachment_context, image_data_urls = _prepare_attachments(payload.attachments)
            turns, _ = ASSISTANT_MEMORY.load(current_user.id)
            memory_context = ASSISTANT_MEMORY.build_context(turns)
            activity_context = ASSISTANT_MEMORY.load_workspace_context(current_user.id)
            answer = ""
            for chunk in ConversationalAssistantAgent(llm).stream_respond(
                payload.message,
                memory_context,
                attachment_context=attachment_context,
                image_data_urls=image_data_urls,
                activity_context=activity_context,
            ):
                if not isinstance(chunk, str) or not chunk:
                    continue
                answer += chunk
                yield _assistant_sse({"type": "delta", "text": chunk})
            if not answer.strip():
                raise RuntimeError("The LLM provider returned an empty response.")
            memory_turn_count = ASSISTANT_MEMORY.remember_exchange(
                current_user.id,
                payload.message,
                answer,
            )
            yield _assistant_sse({
                "type": "complete",
                "memory_turn_count": memory_turn_count,
            })
        except Exception as error:
            LOGGER.exception("Assistant streamed response failed for user %s", current_user.id)
            yield _assistant_sse({
                "type": "error",
                "message": f"The assistant could not complete that reply. {describe_provider_error(error)}",
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/assistant/session")
def clear_assistant_session(
    current_user: AuthUser = Depends(get_current_user),
) -> dict:
    ASSISTANT_MEMORY.clear(current_user.id)
    return {"cleared": True}


@app.post("/api/swarm/query", response_model=QueryResponse)
def run_swarm_query(
    payload: QueryRequest,
    _current_user: AuthUser = Depends(get_current_user),
) -> QueryResponse:
    model = build_llm_adapter().model
    cache_hit = None if _bypass_answer_cache(payload.query, bool(payload.attachments)) else ANSWER_CACHE.lookup(
        _current_user.id,
        model,
        payload.query,
    )
    activity_id = ACTIVITY_STORE.record(
        _current_user.id,
        payload.query,
        model,
        attachment_count=len(payload.attachments),
        cache_hit=cache_hit is not None,
    )
    if cache_hit:
        for step in cache_hit.result.get("steps", []):
            ACTIVITY_STORE.record_step(activity_id, _current_user.id, step["role"], step.get("output", ""))
        ACTIVITY_STORE.record_insights(activity_id, _current_user.id, cache_hit.result)
        return _query_response(payload.query, cache_hit.result, cache_hit)

    engine = ReasoningSwarmEngine(total_agents=payload.total_agents)
    if not engine.llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured. Set GROQ_API_KEY.")

    try:
        workflow = engine.execute_workflow(payload.query, user_id=_current_user.id)
    except WorkflowExecutionError as error:
        raise HTTPException(
            status_code=502,
            detail=f"LLM workflow failed during {error.stage}: {error}",
        ) from error

    _attach_evidence_ledger(workflow, _current_user.id, payload.query, engine.llm)
    for step in workflow["steps"]:
        ACTIVITY_STORE.record_step(activity_id, _current_user.id, step["role"], step.get("output", ""))
    ACTIVITY_STORE.record_insights(activity_id, _current_user.id, workflow)
    if not _bypass_answer_cache(payload.query, bool(payload.attachments)) and not workflow.get("route", {}).get("requires_web_search"):
        ANSWER_CACHE.store(_current_user.id, workflow["model"], payload.query, workflow)
    return _query_response(payload.query, workflow)


def _prepare_attachments(attachments: list[AttachmentRequest]) -> tuple[str, list[str]]:
    text_parts = []
    image_data_urls = []
    remaining_text_chars = 100_000
    allowed_images = {"image/jpeg", "image/png", "image/webp"}
    allowed_text_extensions = {
        ".txt", ".md", ".csv", ".json", ".py", ".js", ".jsx", ".ts", ".tsx",
        ".html", ".css", ".xml", ".yaml", ".yml", ".log", ".sql", ".sh",
    }

    for attachment in attachments:
        extension = PurePath(attachment.name).suffix.lower()
        if attachment.media_type in allowed_images:
            expected_prefix = f"data:{attachment.media_type};base64,"
            if not attachment.content.startswith(expected_prefix):
                raise HTTPException(status_code=400, detail=f"Invalid image data for {attachment.name}.")
            try:
                image_bytes = base64.b64decode(attachment.content[len(expected_prefix):], validate=True)
            except (binascii.Error, ValueError) as error:
                raise HTTPException(status_code=400, detail=f"Invalid image data for {attachment.name}.") from error
            if len(image_bytes) > 5_000_000:
                raise HTTPException(status_code=413, detail=f"{attachment.name} exceeds the 5 MB file limit.")
            image_data_urls.append(attachment.content)
            continue

        if extension == ".pdf" and attachment.media_type == "application/pdf":
            expected_prefix = "data:application/pdf;base64,"
            if not attachment.content.startswith(expected_prefix):
                raise HTTPException(status_code=400, detail=f"Invalid PDF data for {attachment.name}.")
            try:
                pdf_bytes = base64.b64decode(attachment.content[len(expected_prefix):], validate=True)
                if len(pdf_bytes) > 5_000_000:
                    raise HTTPException(status_code=413, detail=f"{attachment.name} exceeds the 5 MB file limit.")
                from pypdf import PdfReader

                reader = PdfReader(BytesIO(pdf_bytes))
                if len(reader.pages) > 50:
                    raise HTTPException(status_code=413, detail=f"{attachment.name} exceeds the 50 page limit.")
                extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except HTTPException:
                raise
            except Exception as error:
                raise HTTPException(status_code=400, detail=f"Could not read {attachment.name} as a PDF.") from error
            text = extracted_text[:remaining_text_chars]
            remaining_text_chars -= len(text)
            text_parts.append(f"FILE: {attachment.name}\n{text}")
            continue

        if extension not in allowed_text_extensions:
            raise HTTPException(status_code=415, detail=f"File type is not supported: {attachment.name}.")
        text = attachment.content[:remaining_text_chars]
        remaining_text_chars -= len(text)
        text_parts.append(f"FILE: {attachment.name}\n{text}")

    return "\n\n".join(text_parts), image_data_urls


def _stream_workflow(
    query: str,
    total_agents: int,
    user_id: int,
    attachments: list[AttachmentRequest] | None = None,
) -> StreamingResponse:
    attachment_context, image_data_urls = _prepare_attachments(attachments or [])
    model = build_llm_adapter().model
    cache_hit = None if _bypass_answer_cache(query, bool(attachments)) else ANSWER_CACHE.lookup(user_id, model, query)
    activity_id = ACTIVITY_STORE.record(
        user_id,
        query,
        model,
        attachment_count=len(attachments or []),
        cache_hit=cache_hit is not None,
    )

    if cache_hit:
        cached_workflow = cache_hit.result
        for step in cached_workflow.get("steps", []):
            ACTIVITY_STORE.record_step(activity_id, user_id, step["role"], step.get("output", ""))
        ACTIVITY_STORE.record_insights(activity_id, user_id, cached_workflow)
        cache_event = {
            "type": "cache_hit",
            "answer": cached_workflow["answer"],
            "steps": cached_workflow["steps"],
            "selected_agents": cached_workflow["selected_agents"],
            "action_probabilities": cached_workflow["action_probabilities"],
            "validation_score": cached_workflow["validation_score"],
            "validation_issues": cached_workflow["validation_issues"],
            "latency_ms": 0,
            "model": cached_workflow["model"],
            "similarity": round(cache_hit.similarity, 4),
            "matched_query": cache_hit.query,
        }

        def cached_event_stream():
            yield f"data: {json.dumps(cache_event, ensure_ascii=True)}\n\n"

        return StreamingResponse(
            cached_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    engine = ReasoningSwarmEngine(total_agents=total_agents)
    if not engine.llm.enabled:
        raise HTTPException(status_code=503, detail="Live LLM is not configured. Set GROQ_API_KEY.")

    def event_stream():
        if attachment_context or image_data_urls:
            events = engine.iter_workflow(
                query,
                attachment_context=attachment_context,
                image_data_urls=image_data_urls,
                user_id=user_id,
            )
        else:
            events = engine.iter_workflow(query, user_id=user_id)
        for event in events:
            if event.get("type") == "step" and activity_id is not None:
                ACTIVITY_STORE.record_step(
                    activity_id,
                    user_id,
                    event["role"],
                    event.get("output", ""),
                )
            if event.get("type") == "complete" and activity_id is not None:
                ACTIVITY_STORE.record_insights(activity_id, user_id, event)
            if event.get("type") == "complete":
                _attach_evidence_ledger(event, user_id, query, engine.llm)
            if (
                event.get("type") == "complete"
                and not _bypass_answer_cache(query, bool(attachments))
                and not event.get("route", {}).get("requires_web_search")
            ):
                ANSWER_CACHE.store(user_id, engine.llm.model, query, event)
            yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/swarm/query/stream")
def stream_swarm_query(
    payload: QueryRequest,
    _current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    return _stream_workflow(payload.query, payload.total_agents, _current_user.id, payload.attachments)


@app.get("/api/swarm/query/stream")
def stream_swarm_query_get(
    query: str = Query(..., min_length=1, max_length=2000),
    total_agents: int = Query(default=10_000, ge=1, le=100_000),
    _current_user: AuthUser = Depends(get_current_user),
) -> StreamingResponse:
    return _stream_workflow(query, total_agents, _current_user.id)


@app.get("/")
def index() -> dict:
    return {
        "message": "Swarm Production API is running.",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "/api/swarm/query",
        "stream_endpoint": "/api/swarm/query/stream",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
