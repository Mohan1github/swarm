# Groq Cloud and Live Workflow

The API and desktop/terminal apps use Groq's OpenAI-compatible endpoint and `openai/gpt-oss-120b` by default. Reasoning queries run eleven model stages: planner, researcher, analyst, critic, executor, synthesizer draft, corrector, synthesizer revision, query focus, response summarizer, and validator. If the classifier determines current web evidence is needed, a separate web retriever stage runs before the researcher; this adds no LLM call. Evidence-sensitive web research then adds one claim-level evidence audit model request. Including query classification, a reasoning request without web research makes twelve model requests; one with web research and the evidence audit makes thirteen. Direct-routed queries retain the classifier-plus-answer path and never invoke web search. The 10,000 local workers remain lightweight selection/scoring objects; they do not each make a separate model request.

Create a local `.env` file (ignored by Git) and set your own Groq key. Never commit or share the key:

```dotenv
GROQ_API_KEY=your-groq-api-key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
JWT_SECRET_KEY=replace-with-a-long-random-secret
TAVILY_API_KEY=your-tavily-api-key
# Optional: set to a vision-capable model available to your Groq account to analyze photos.
GROQ_VISION_MODEL=your-vision-capable-groq-model
```

The API provides `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, and `POST /api/auth/logout`. New passwords must contain at least 10 characters. Access tokens expire after eight hours; logout revokes the account's active tokens. All `/api/swarm/query` endpoints require `Authorization: Bearer <access_token>`. The account database is stored in `swarm_auth.sqlite3` locally and in the `swarm-auth-data` Docker volume under Compose.

Start the API with Docker Compose:

```powershell
docker compose up --build
```

For local Python use, the adapter automatically loads the same `.env` file. Alternatively, set the key in PowerShell and start the API:

```powershell
$env:GROQ_API_KEY = "your-groq-api-key"
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## API

`DELETE /api/history` clears the authenticated user's Swarm activity, Assistant conversation and workspace context, cached answers, stored agent-output vectors, web research sources, and Evidence Ledger reports. It does not affect other users.

`GET /api/evidence-ledger?limit=20` returns the authenticated user's latest saved claim audits (maximum 100). For web-researched reasoning responses, `POST /api/swarm/query` returns an `evidence_ledger`; the completion event on `POST /api/swarm/query/stream` includes the same field, with claim verdicts, source links, matched passages, publication/fetch dates, and a report ID. If the auditor or persistence step fails, the answer still completes and the response includes `evidence_ledger_error`. The auditor checks source-text support; it does not determine objective truth, and a supported verdict is not a guarantee of correctness. Set `EVIDENCE_LEDGER_DB_PATH` to choose another SQLite location; Docker Compose stores it under `/data`.

The Assistant reviews only the signed-in user's saved conversation and private activity graph when the user opens Assistant mode, then streams a short continuity-aware greeting. If the user remains idle, it can stream up to two low-pressure, context-grounded follow-ups per open session, after a 45-second idle period; the UI has a pause/resume control. Completed proactive messages are stored in that user's conversation so they are not repeated without context. Replies stream as generated and are saved only after completion. This is limited to the active app session, not background monitoring; it does not initiate web searches or contact the user outside the app. The assistant is designed to sound warm and attentive without claiming to be human or to have feelings. Use `POST /api/assistant/session/stream` to open a streamed session, `POST /api/assistant/turn/stream` to send a message, and `POST /api/assistant/proactive/stream` for a proactive turn; all require authentication and return Server-Sent Events. The existing non-streaming session and turn endpoints remain available.

`POST /api/swarm/query` runs all stages and returns the final answer, model-derived action probabilities, per-stage outputs and durations, total latency, and the validator's self-assessment. The validation score is not a guarantee of factual correctness. Requests return `503` when no provider key is configured and `502` when a stage fails; the server does not substitute a simulation answer.

```json
{
	"query": "Explain a safe deployment rollout"
}
```

`GET /api/swarm/query/stream?query=...` sends Server-Sent Events and requires an `Authorization: Bearer <access_token>` header. Use an authenticated API client; the web client uses the POST stream so it can attach credentials and files.

`POST /api/swarm/query/stream` sends the same events to API clients. Example from PowerShell:

```powershell
curl.exe -N -X POST http://localhost:8000/api/swarm/query/stream `
	-H "Content-Type: application/json" `
	-d '{"query":"Explain a safe deployment rollout"}'
```

The web chat accepts up to five attachments per message, each up to 5 MB, with a combined 15 MB payload limit: PNG/JPEG/WebP photos, PDFs, and common text, data, and source files. PDFs are text-extracted; photos are sent to `GROQ_VISION_MODEL` (or `GROQ_MODEL` if the latter is vision-capable).

Successful text-only workflows are cached per authenticated user and model in a persistent SQLite vector index. The default similarity threshold is `0.93` and can be changed with `VECTOR_CACHE_SIMILARITY_THRESHOLD`; cache hits emit one `cache_hit` stream event and bypass agent construction. Requests with attachments are not cached. The cache uses local hashed lexical features, so it is optimized for repeated or closely reworded queries rather than broad semantic equivalence. Docker Compose stores it in the existing `swarm-auth-data` volume.

On live reasoning workflows, each completed stage output is stored as a per-user hashed lexical feature vector in `agent_output_vectors.sqlite3` (up to 5,000 outputs per user). `QueryFocusAgent` extracts the exact request, constraints, and comparisons; `AgentOutputVectorAgent` retrieves related outputs from the current and prior runs; `ResponseSummarizerAgent` combines those matches with the current run's stage outputs before the validator assesses the final response. This separate output index is not the answer cache and does not use a hosted embedding API. Set `AGENT_OUTPUT_VECTOR_DB_PATH` to change its local SQLite path. Direct-routed requests and answer-cache hits retain their existing fast paths.

The classifier sets `requires_web_search` only for reasoning requests needing current/changing information, current news, or explicit factual verification; it does not search for basic facts, code help, private/personal details, credentials, or attachment content. Configure `TAVILY_API_KEY` to enable it. LangGraph orchestrates a Tavily text-only search and a local FastEmbed (`BAAI/bge-small-en-v1.5`) index/retrieval flow. The first web lookup downloads the embedding model; no image search or binary payload is stored. Query text is sent to Tavily, so do not include personal/private details. Search results are untrusted evidence, not a guarantee of truth: the summarizer is instructed to cite source URLs/dates, seek corroboration, and report conflicts.

Web source text, dense vectors, conversation-topic centroids, and weighted nearest-neighbor links are stored in the user-scoped `web_research.sqlite3` database. Retrieval combines BM25, dense cosine ranking, reciprocal-rank fusion, and Dijkstra traversal over the user's source graph. The database persists across sessions for that user's related topics; `DELETE /api/history` clears that user's web research records too. Set `WEB_RESEARCH_DB_PATH` to choose another SQLite location. Docker Compose keeps the downloaded embedding model in `/data/fastembed` across restarts. Web research requests bypass the answer cache to avoid stale cached results.

`GET /api/activity/graph` returns the authenticated user's recent prompt and agent-stage graph. It stores prompt text, metadata, and excerpts of at most 320 characters from each streamed stage output so the graph can show generated work; full final answers and uploaded file contents are not stored in the activity log. It returns up to 80 prompts and 160 recent stage events, plus related-prompt and prompt-to-stage edges.

The interactive desktop app and terminal app use the same live stage workflow. They require `GROQ_API_KEY`; neither returns a canned answer when the provider is unavailable.