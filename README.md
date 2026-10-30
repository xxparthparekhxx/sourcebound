# docgraph

A document assistant that answers questions over a set of local documents and cites its sources.
It runs fully locally with [Ollama](https://ollama.com), or against any OpenAI-compatible endpoint.
"docgraph" is a working name.

- **Ingest** PDF (with page numbers), Markdown, plain text and HTML. Chunking follows headings and
  paragraphs. Re-ingesting a changed file replaces its chunks.
- **Retrieve** with vector search (MMR) and BM25 fused by reciprocal rank fusion, optionally expanded
  through a small knowledge graph and reranked by [openjev-flash](https://huggingface.co/xxparthparekhxx/openjev-flash).
  Every stage can be switched on or off.
- **Answer** in `pipeline` mode (one call over the retrieved passages) or `agent` mode (the model calls
  `search_documents`, `read_passage` and `related_entities`). Answers stream, every `[n]` citation is
  checked against the passages the model was actually given, and the assistant says so instead of
  answering when nothing relevant is found.
- **Use** it from a CLI, a FastAPI app (SSE streaming) or the single HTML page the app serves.

```
                 ┌──────────────────────── ingest ────────────────────────┐
 pdf / md /      │ load ─► structure-aware ─► embed ─► ChromaDB (vectors) │
 txt / html ────►│ (OCR   chunks (document,   (local │                    │
                 │ opt.)   page, heading)     or API)├► SQLite FTS5 (BM25)│
                 │                                   └► SQLite graph      │
                 │          optional: LLM entity/relation extraction ─────┘
                 └─────────────────────────────────────────────────────────┘

 question ─► vector (+MMR) ─┐
             BM25 ──────────┼─► RRF ─► [rerank] ─► top-k passages ─► relevant? ─no─► "nothing relevant"
             graph expansion┘                                           │yes
                                                                         ▼
             pipeline: one streamed call        agent: tool calls (search_documents, read_passage,
             over the numbered passages         related_entities), step cap; falls back to pipeline
                                  └──────────► citation check: [n] must be a passage the model was given
```

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repository> && cd docgraph
uv sync                               # add `--extra rerank` for openjev-flash, `--extra eval` for the evaluation
cp .env.example .env                  # only needed for endpoints that use API keys
```

### With Ollama (fully local)

Any chat model works; `bonsai-2` is what this was developed against. The default configuration points
at `http://localhost:11434/v1`, so no config file is needed:

```bash
ollama pull bonsai-2
uv run docgraph ingest ./my-documents          # first run downloads a ~130 MB embedding model
uv run docgraph ask "What optimizer was used, and with which hyperparameters?"
uv run docgraph serve                          # http://127.0.0.1:8000
```

Two Ollama details that matter:

- **Context length.** Ollama truncates prompts to its context length (4096 by default) without
  telling the client. Set `context_tokens` on the endpoint to the real value (the example config does,
  for 4096). docgraph then drops the lowest-ranked passages to fit and never sends a prompt that would
  be cut off; if even one passage does not fit, it fails with an error. To use more context, start the
  server with `OLLAMA_CONTEXT_LENGTH=16384` and raise `context_tokens` to match.
- **Thinking models.** `bonsai-2` thinks by default. `options = { reasoning_effort = "none" }` on the
  endpoint turns that off through Ollama's OpenAI-compatible API. `options` is sent with every request
  to that endpoint, so it is also where sampling settings go.

### With any OpenAI-compatible endpoint

```bash
export OPENAI_API_KEY=...                      # the key stays in the environment, never in a file
export DOCGRAPH_BASE_URL=https://api.openai.com/v1
export DOCGRAPH_MODEL=gpt-4o-mini
export DOCGRAPH_API_KEY_ENV=OPENAI_API_KEY
uv run docgraph ingest ./my-documents
uv run docgraph ask "..."
```

For several endpoints with fallback, copy [`docgraph.example.toml`](docgraph.example.toml) to
`docgraph.toml`. Endpoints are tried in order; a connection error, timeout, HTTP 429 or 5xx moves on to
the next one, and an endpoint whose key variable is unset is skipped. Other HTTP errors (a bad key,
an unknown model) are reported, not retried elsewhere. A streamed answer can only fall back before its
first token.

```toml
[[chat.endpoints]]
name = "nim"
base_url = "https://integrate.api.nvidia.com/v1"
api_key_env = "NVIDIA_API_KEY"
model = "meta/llama-3.3-70b-instruct"

[[chat.endpoints]]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "nvidia/nemotron-3.5-super-vl:free"

[[chat.endpoints]]
name = "ollama"
base_url = "http://localhost:11434/v1"
model = "bonsai-2"
context_tokens = 4096
options = { reasoning_effort = "none" }
```

Embeddings are configured separately (`[embeddings]`): `provider = "local"` runs a small CPU model
through fastembed (default `BAAI/bge-small-en-v1.5`), `provider = "openai"` uses any `/v1/embeddings`
endpoint. The index records the embedding model name and dimension, and refuses to be opened with a
different one (it tells you to use a new `data_dir` or run `docgraph reset`).

API keys are read from environment variables only; a config file that contains `api_key = ...` is
rejected. `.env` in the working directory is loaded without overriding variables that are already set.

## Commands

```
docgraph ingest <path> [--graph] [--ocr] [--force]   index a file or directory
docgraph ask "<question>" [--mode pipeline|agent] [--top-k N] [--no-vector] [--no-bm25] [--no-mmr]
                          [--graph] [--rerank] [--json]
docgraph docs | remove <path> | reset --yes          inspect and manage the index
docgraph serve [--host 127.0.0.1] [--port 8000]
docgraph eval prepare|retrieval|answers|dev|report   the Qasper evaluation (needs a source checkout)
```

- `ingest` skips unchanged files (SHA-256) and replaces a changed file's chunks, vectors and graph rows.
  Pages of a PDF with no text layer are reported as skipped. `--ocr` sends those pages to a vision-capable
  model on the configured endpoint (`[ingest.ocr]`, off by default).
- `--graph` makes one LLM call per chunk to extract entities and relations as JSON. Replies are validated
  against a schema and dropped when they do not fit. This is slow on a small local model (see the
  evaluation notes), which is why it is a flag.

HTTP API (`docgraph serve`):

| endpoint | |
|---|---|
| `POST /ask` | body `{"question", "mode"?, "top_k"?, "vector"?, "bm25"?, "mmr"?, "graph"?, "rerank"?}`. Server-sent events: `retrieval`, `token`*, (`tool`, `retract`, `notice` in agent mode), then `citations` (each with `n`, `id`, `document`, `file`, `page`, `heading`, `text`; plus `flagged` and the passages shown to the model), then `done`. Failures arrive as an `error` event. |
| `GET /passages/{id}` | one passage with document, file, page, heading, text and the ids of the previous and next passage |
| `GET /healthz` | status, index counts, embedding model, chat endpoints (name, model, usable) |
| `GET /` | the single-page UI: question box, streaming answer, clickable citations that open the passage |

`create_app(..., ask_dependencies=[...])` accepts FastAPI dependencies that run before every `/ask`
request, which is where rate limits or a bot check would go.

## How it works

**Chunking.** Markdown and HTML are split at headings and paragraphs; PDFs use PyMuPDF text blocks
with heading detection by font size and weight, and running headers and footers are dropped. Chunks
of up to 1200 characters are packed from whole paragraphs, never cross a heading, and carry about 200
characters of overlap with the previous chunk. Each chunk records document, page(s) and the nearest
heading. The embedded text is prefixed with `title > heading`.

**Retrieval.** Vector candidates come from ChromaDB (cosine); with MMR on, 40 are fetched and 20 are
chosen with lambda 0.7. BM25 comes from SQLite FTS5 (porter stemming; text, heading and title columns).
The lists are fused with reciprocal rank fusion (k = 60). Graph expansion matches entity names in the
question, then adds chunks that mention those entities and chunks that mention their one-hop neighbours,
weighting rare entities higher. Reranking scores each candidate passage against the question with
openjev-flash (P(entailment) of "the passage answers the question ...").

**"Nothing relevant".** Retrieval reports relevant when the best vector similarity reaches
`min_vector_score` (0.55), or a returned passage contains at least half of the question's content words
(and at least two of them), or the question names a graph entity. Otherwise `ask` answers with a fixed
message and makes no model call. The prompt also tells the model to say so when the passages lack the
answer.

**Citations.** Passages are numbered `[1]..[k]` in the prompt. A filter on the token stream holds back
each `[...]` until it closes, and removes any number that is not a passage the model was given (a
partly valid `[1, 9]` becomes `[1]`). Removed numbers are reported as `flagged`.

**Agent mode.** The model gets `search_documents`, `read_passage` and `related_entities` through native
tool calling, at most `answer.max_steps` rounds; after that it must answer from what it has. Passages are
numbered as the tools return them. If the endpoint rejects tools, a tool call is malformed (bad JSON, an
unknown tool, a missing or mistyped argument), or the model answers without searching, docgraph falls back
to pipeline mode automatically and says so (`notice` event, `fallback` in `done`).

EVAL_AND_TESTED_PLACEHOLDER

## Limits and what is not done

LIMITS_PLACEHOLDER

## Development

```bash
uv sync --extra rerank --extra eval
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 uv run pytest   # no network, no model downloads
```

Source layout: `src/docgraph/` (`config`, `llm`, `embeddings`, `ingest/`, `store`, `graph`, `retrieval`,
`rerank`, `answer`, `agent`, `citations`, `api`, `cli`, `static/index.html`), `tests/`, `eval/`.

MIT licensed, see [LICENSE](LICENSE).
