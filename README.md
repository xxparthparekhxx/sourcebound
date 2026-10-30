# sourcebound

A document assistant that answers questions over a set of local documents and cites its sources.
It runs fully locally with [Ollama](https://ollama.com), or against any OpenAI-compatible endpoint.

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
git clone <this repository> && cd sourcebound
uv sync                               # add `--extra rerank` (torch) or `--extra rerank-onnx` (ONNX Runtime) for openjev-flash, `--extra eval` for the evaluation
cp .env.example .env                  # only needed for endpoints that use API keys
```

### With Ollama (fully local)

Any chat model works; `bonsai-2` is what this was developed against. The default configuration points
at `http://localhost:11434/v1`, so no config file is needed:

```bash
ollama pull bonsai-2
uv run sourcebound ingest ./my-documents          # first run downloads a ~130 MB embedding model
uv run sourcebound ask "What optimizer was used, and with which hyperparameters?"
uv run sourcebound serve                          # http://127.0.0.1:8000
```

Two Ollama details that matter:

- **Context length.** Ollama truncates prompts to its context length (4096 by default) without
  telling the client. Set `context_tokens` on the endpoint to the real value (the example config does,
  for 4096). sourcebound then drops the lowest-ranked passages to fit and never sends a prompt that would
  be cut off; if even one passage does not fit, it fails with an error. The estimate is rough, so when the server reports far fewer prompt tokens than the prompt contains, the answer gets a warning notice. To use more context, start the
  server with `OLLAMA_CONTEXT_LENGTH=16384` and raise `context_tokens` to match.
- **Thinking models.** `bonsai-2` thinks by default. `options = { reasoning_effort = "none" }` on the
  endpoint turns that off through Ollama's OpenAI-compatible API. `options` is sent with every request
  to that endpoint, so it is also where sampling settings go.

### With any OpenAI-compatible endpoint

```bash
export OPENAI_API_KEY=...                      # the key stays in the environment, never in a file
export SOURCEBOUND_BASE_URL=https://api.openai.com/v1
export SOURCEBOUND_MODEL=gpt-4o-mini
export SOURCEBOUND_API_KEY_ENV=OPENAI_API_KEY
uv run sourcebound ingest ./my-documents
uv run sourcebound ask "..."
```

For several endpoints with fallback, copy [`sourcebound.example.toml`](sourcebound.example.toml) to
`sourcebound.toml`. Endpoints are tried in order; a connection error, timeout, HTTP 429 or 5xx moves on to
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
different one (it tells you to use a new `data_dir` or run `sourcebound reset`).

openjev-flash (rerank and faithfulness) runs on either backend: `[rerank] backend = "torch"` (default, `--extra rerank`)
or `backend = "onnx"` (`--extra rerank-onnx`), which runs the int8 export under `onnx/` in the model repository with
ONNX Runtime and needs no PyTorch. `revision` pins the model repository to a commit, `threads` limits the ONNX
Runtime threads, and `onnx_dir` points at a local copy of the export instead of downloading it.

API keys are read from environment variables only; a config file that contains `api_key = ...` is
rejected. `.env` in the working directory is loaded without overriding variables that are already set.

## Commands

```
sourcebound ingest <path> [--graph] [--ocr] [--force]   index a file or directory
sourcebound ask "<question>" [--mode pipeline|agent] [--top-k N] [--no-vector] [--no-bm25] [--no-mmr]
                          [--graph] [--rerank] [--json]
sourcebound docs | remove <path> | reset --yes          inspect and manage the index
sourcebound serve [--host 127.0.0.1] [--port 8000]
sourcebound eval prepare|retrieval|answers|dev|report   the Qasper evaluation (needs a source checkout)
```

- `ingest` skips unchanged files (SHA-256) and replaces a changed file's chunks, vectors and graph rows.
  Pages of a PDF with no text layer are reported as skipped. `--ocr` sends those pages to a vision-capable
  model on the configured endpoint (`[ingest.ocr]`, off by default).
- `--graph` makes one LLM call per chunk to extract entities and relations as JSON. Replies are validated
  against a schema and dropped when they do not fit. This is slow on a small local model (see the
  evaluation notes), which is why it is a flag.

HTTP API (`sourcebound serve`):

| endpoint | |
|---|---|
| `POST /ask` | body `{"question", "mode"?, "top_k"?, "vector"?, "bm25"?, "mmr"?, "graph"?, "rerank"?}`. Server-sent events: `retrieval`, `token`*, (`tool`, `retract`, `notice` in agent mode), then `citations` (each with `n`, `id`, `document`, `file`, `page`, `heading`, `text`; plus `flagged` and the passages shown to the model), then `done`. Failures arrive as an `error` event. |
| `GET /passages/{id}` | one passage with document, file, page, heading, text and the ids of the previous and next passage |
| `GET /healthz` | status, index counts, embedding model, chat endpoints (name, model, usable) |
| `GET /` | the single-page UI: question box, streaming answer, clickable citations that open the passage |

`sourcebound serve --public` is a locked-down variant for a public demo behind a gateway: every route except
`GET /healthz` (which answers `{"status": "ok"}` and nothing else) needs an `X-Gateway-Secret` header equal to
the environment variable named by `[serve].gateway_secret_env`, otherwise 401; the server refuses to start
without it. `POST /ask` takes only `{"question"}` (at most `[serve].max_question_chars` = 500 characters) and
streams `retrieval`, `token`*, `citations` (with passage text), `faithfulness` (each cited sentence scored by
openjev-flash against the passages it cites) and `done` (with timings); at most `[serve].max_concurrent` = 2
questions run at once and the next gets 503. Failures arrive as one generic `error` event.

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
`min_vector_score` (0.6), or one returned passage contains all of the question's content words (at
least two of them unless the question has only one), or the question names a graph entity. Otherwise `ask` answers with a fixed
message and makes no model call. The prompt also tells the model to say so when the passages lack the
answer.

**Citations.** Passages are numbered `[1]..[k]` in the prompt. A filter on the token stream holds back
each `[...]` until it closes, and removes any number that is not a passage the model was given (a
partly valid `[1, 9]` becomes `[1]`). Removed numbers are reported as `flagged`.

**Agent mode.** The model gets `search_documents`, `read_passage` and `related_entities` through native
tool calling, at most `answer.max_steps` rounds; after that it must answer from what it has. Passages are
numbered as the tools return them. If the endpoint rejects tools, a tool call is malformed (bad JSON, an
unknown tool, a missing or mistyped argument), or the model answers without searching, sourcebound falls back
to pipeline mode automatically and says so (`notice` event, `fallback` in `done`).

## Evaluation

Everything below is produced by scripts in `eval/` and can be regenerated; the raw per-question JSON is in
[`eval/results/`](eval/results) and [`eval/results/SUMMARY.md`](eval/results/SUMMARY.md) is written by
`sourcebound eval report`. Numbers are copied from those files. All runs are from 2026-10-30.

**Data.** [Qasper](https://huggingface.co/datasets/allenai/qasper) (CC BY 4.0), validation split: 40 papers and the
176 answerable questions with annotated evidence that belong to them, converted to Markdown (title, abstract, sections,
captions). 10 further papers (35 questions), disjoint from the 40, were used to choose the rerank prompt and the
relevance-gate threshold; the tables below use only the 40 held-out papers. Papers are indexed as 1,396 chunks (1,200
characters, 200 overlap) with `BAAI/bge-small-en-v1.5`. A retrieved chunk counts as a hit when at least half of a gold evidence
paragraph's word 5-grams occur in it; 466 of the 483 gold paragraphs are covered by some chunk at all. Recall is the best
over the question's annotators. Cells are the mean with a 95% bootstrap interval over questions.

```bash
sourcebound eval prepare                                   # downloads Qasper, writes eval/data/qasper
sourcebound eval retrieval --build --graph                 # index the papers; --graph runs entity extraction (bonsai-2, 5,186 s)
sourcebound eval retrieval --tag main                      # ablation; settings can be added later with --update
sourcebound eval answers --tag ollama-bonsai2 --endpoint ollama --n 30
sourcebound eval report
```

### Retrieval ablation (40 held-out papers, 176 questions)

*Within the question's paper* (the Qasper protocol: only that paper's chunks are searched):

| setting | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| vector | 0.546 (0.477-0.611) | 0.761 (0.704-0.813) | 0.432 (0.377-0.487) |
| vector + MMR | 0.510 (0.440-0.580) | 0.719 (0.659-0.775) | 0.418 (0.362-0.476) |
| BM25 | 0.496 (0.430-0.564) | 0.710 (0.648-0.767) | 0.374 (0.319-0.430) |
| hybrid (vector + MMR, BM25, RRF) | 0.586 (0.521-0.652) | 0.752 (0.696-0.805) | 0.450 (0.397-0.504) |
| hybrid + graph | 0.562 (0.494-0.627) | 0.740 (0.680-0.795) | 0.409 (0.358-0.463) |
| hybrid + rerank | 0.629 (0.566-0.696) | 0.801 (0.750-0.851) | 0.449 (0.397-0.503) |
| hybrid + graph + rerank | 0.633 (0.568-0.698) | 0.810 (0.760-0.857) | 0.450 (0.398-0.505) |

*Pooled over all 40 papers* (what the assistant does with a document set):

| setting | recall@5 | recall@10 | MRR@10 |
|---|---|---|---|
| vector | 0.243 (0.183-0.304) | 0.355 (0.289-0.422) | 0.181 (0.137-0.226) |
| vector + MMR | 0.201 (0.144-0.260) | 0.289 (0.228-0.356) | 0.156 (0.113-0.202) |
| BM25 | 0.255 (0.194-0.320) | 0.343 (0.273-0.411) | 0.174 (0.130-0.221) |
| hybrid | 0.331 (0.267-0.399) | 0.399 (0.331-0.466) | 0.205 (0.158-0.252) |
| hybrid + graph | 0.284 (0.220-0.350) | 0.405 (0.338-0.474) | 0.193 (0.146-0.239) |
| hybrid + rerank | 0.329 (0.264-0.397) | 0.427 (0.358-0.499) | 0.236 (0.185-0.287) |
| hybrid + graph + rerank | 0.328 (0.263-0.397) | 0.418 (0.349-0.487) | 0.232 (0.181-0.283) |

Source: `eval/results/retrieval-main.json` (three runs merged with `--update`; the commands are listed in `SUMMARY.md`).
Graph: 3,924 entities and 1,971 relations from `bonsai-2` over the 1,396 chunks (188 replies dropped as invalid; 129 placeholder entities
such as `INLINEFORM0` removed). Rerank: `xxparthparekhxx/openjev-flash` on CPU, hypothesis template chosen on the dev papers.

What the numbers support (paired bootstrap over questions, computed from the same file):

- Hybrid beats BM25 alone (recall@5 +0.09 within a paper, 95% interval +0.03 to +0.15) and pooled it beats both single
  retrievers at recall@5 (+0.09 over vector, interval +0.04 to +0.14). Against vector alone within a paper the difference is
  not distinguishable from zero.
- MMR on the vector side lowered every metric here (vector → vector + MMR); it is on by default in `hybrid`, and the hybrid
  figures include it. Turning it off with `--no-mmr` is worth trying on a corpus like this one.
- The knowledge graph did **not** help on this benchmark: hybrid + graph is lower than hybrid on most metrics (MRR within a
  paper −0.04, interval −0.07 to −0.01). Extraction by a small model is noisy, and Qasper questions are mostly about the one paper
  rather than about entities shared across papers. Graph expansion is off by default.
- Reranking adds roughly +0.04 to +0.05 recall@10 within a paper, but the interval includes zero (−0.00 to +0.10); pooled, the
  recall@10 gain is +0.03 (−0.01 to +0.06). It took 9 to 12 minutes per 176 questions on a loaded CPU, so it is off by default.

### Answer quality (30 questions, 15 for agent mode)

Retrieval restricted to the question's paper, top 5 passages, `bonsai-2` on Ollama unless stated, sample seed 1.

| run | mode | answered / no answer | cites something | citation precision | cited sentences judged supported |
|---|---|---|---|---|---|
| `bonsai-2` (Ollama) | pipeline | 28 / 2 | 24 | 0.348 (16/46 cited passages overlap gold); per-answer mean 0.431 (0.257-0.604) | 0.963 (0.907-1.000) of 54 |
| `bonsai-2` (Ollama) | agent | 15 / 0 | 14 | 0.214 (6/28); per-answer mean 0.274 (0.107-0.476) | 0.738 (0.595-0.857) of 42 |
| `nvidia/nemotron-3.5-super-vl:free` (OpenRouter) | pipeline | 29 / 0 (1 call failed) | 25 | 0.346 (18/52); per-answer mean 0.448 (0.301-0.601) | 0.909 (0.836-0.982) of 55 |

Sources: `eval/results/answers-ollama-bonsai2.json`, `answers-ollama-bonsai2-agent.json`, `answers-openrouter-nemotron.json`.

- **Citation precision** is the share of cited passages that overlap a gold evidence paragraph. It is a lower bound: annotators
  mark one or two paragraphs, and other passages can support an answer too. The gold evidence was among the five passages shown
  to the model in only 57% (pipeline) and 47% (agent) of the questions (`SUMMARY.md`), which caps what citation precision can reach.
- **Faithfulness** is the share of answer sentences carrying a `[n]` marker that `openjev-flash` scores as entailed
  (probability ≥ 0.5) by the passages they cite. Sentences without a citation are not scored (22 of 76 in the pipeline run, 29 of 71 in the agent run). It is
  one small model's judgement, not a human one, and the intervals are wide at these sample sizes.
- The pipeline run was done twice (an earlier run, commit `7c548ec`, gave 0.349 and 0.935 on the same questions); sampling
  temperature is 0.2, so repeat runs differ by a few points. No other metric was run twice.
- Agent mode with a 9B ternary model searched on every question and never needed the pipeline fallback, but it cited less precisely
  and wrote less faithful sentences than pipeline mode. On 15 questions that is a difference to watch, not a conclusion.
- The OpenRouter run used 33 HTTP requests (cap 60) and one question failed after three empty streams; it is the only full evaluation against a hosted endpoint.

### Relevance gate

Calibrated on the dev papers (`eval/results/gate-calibration.json`): with `min_vector_score = 0.6` or lexical overlap, 97% of
answerable dev questions pass, 94% of questions about papers *not* in the index also pass (they are NLP questions, close enough to the indexed papers), and 0% of 40 general-knowledge questions pass. On the held-out index it passes 100% of answerable,
100% of out-of-corpus and 5% of off-topic questions. It reliably rejects off-topic questions; it does not detect a missing document.

## Tested with

Only what was actually run (2026-10-30, Linux, Python 3.12):

| component | what was run |
|---|---|
| Chat, local | Ollama 0.40.13 at `http://localhost:11434/v1`, `bonsai-2` (Prism ML, ternary 9B, 3.2 GB) on a laptop GPU, `reasoning_effort = "none"`, native tool calling. Used for pipeline and agent answers, graph extraction, the CLI, the HTTP API and the evals. Ollama's context length was left at its default of 4,096. |
| Chat, hosted | OpenRouter (`https://openrouter.ai/api/v1`) with `nvidia/nemotron-3.5-super-vl:free`, pipeline mode: the 30-question answer eval and one CLI `ask` through the `SOURCEBOUND_BASE_URL` shortcut. |
| Fallback | A dead endpoint (connection refused) listed before Ollama: the answer came from Ollama. A 429 on the first endpoint is covered only by unit tests against a local fake server. |
| Embeddings | `BAAI/bge-small-en-v1.5` through fastembed on CPU. `provider = "openai"` embeddings were exercised only against the fake server in the tests. |
| Rerank / faithfulness | `xxparthparekhxx/openjev-flash`, CPU torch. |
| Interfaces | CLI `ingest`, `ask` (both modes), `docs`, `serve`; HTTP `/healthz`, `/ask` (SSE) and `/` against a three-paper index. The HTML page was served but not driven in a browser. |
| Documents | Markdown (the Qasper corpus). PDF, HTML and text loaders are covered by unit tests on small generated files only. |

Not run against: NVIDIA NIM (it appears in the example config), OpenAI, or any other hosted endpoint; OCR with a real vision
model; large PDFs; any corpus other than Qasper.

Tests: `uv run pytest` — 136 tests, no network, no model downloads (a local fake OpenAI-compatible server and a hashing embedder).
They cover chunking, RRF fusion, citation validation, endpoint fallback order (429, 5xx and connection errors move on; other 4xx do
not), agent → pipeline fallback, the embedding-model guard, the context-length checks and the eval metrics.

## Limits and what is not done

- **Context length is the caller's responsibility.** Ollama's OpenAI-compatible API cannot raise the context length per request
  and truncates silently. sourcebound trims passages to the `context_tokens` you declare (using a 3.5 characters-per-token estimate)
  and refuses prompts that do not fit. As a second line of defence, if the server's reported `prompt_tokens` is far below what the
  prompt must contain, the answer carries a `Warning:` notice and `truncated` is set. That check needs the server's usage report
  and a declared `context_tokens`; it cannot catch a truncation of a few tokens.
- **The knowledge graph did not help in the evaluation** (see above) and is slow with a small local model: 5,186 s for 1,396 chunks
  with `bonsai-2`, 13% of replies unusable. Treat it as experimental. There is no entity merging beyond case-insensitive name matching.
- **Evaluation scope.** One dataset, 40 papers, English, single runs; the answer metrics use 30 and 15 questions. Citation
  precision is overlap with annotated evidence and understates correct citations. Faithfulness is judged by a small entailment
  model, and only cited sentences are scored. No human evaluation. Qasper is Markdown here, so the PDF path (page numbers, headings by font size) is not part of the
  numbers.
- **Not verified:** OCR on real scans, PDFs larger than the unit-test files, non-English documents, concurrent users, an OpenAI-compatible
  endpoint other than Ollama and OpenRouter, `provider = "openai"` embeddings against a real server.
- **Agent mode** depends on the endpoint supporting tool calls; otherwise it falls back to pipeline mode. Small models cite less reliably in it (see above).
- **Security.** `sourcebound serve` has no authentication and is meant for localhost; put a proxy in front of it or pass
  `ask_dependencies` to `create_app` before exposing it. Answers are only as trustworthy as the documents: text in a document is
  placed in the prompt and can try to steer the model.
- **Not done:** a React UI, Neo4j or another graph database, analytics, user accounts, multi-user indexes,
  incremental graph updates other than re-extracting pending chunks, a general hosted deployment recipe (the live demo above is one specific deployment).

## Live demo

https://sourcebound.parthparekhdev.com answers questions over the 40 Qasper validation papers used in the evaluation
(1,396 chunks). What runs where:

| Part | Where | What it does |
|---|---|---|
| Page (`web/public/`) | Cloudflare Worker assets | Question box, streamed answer, `[n]` citations that open the passage, a support badge per cited sentence, saved answers. |
| `POST /api/ask` (`web/src/`) | Cloudflare Worker | Checks a Turnstile token, applies the limits below, forwards the question and a secret header to the origin and streams the reply back. |
| Limits | One SQLite-backed Durable Object (free plan) | 10 questions per client per hour (sliding window, client = hash of the IP) and 950 per UTC day for everyone, the free OpenRouter quota being 1,000 requests a day. A question the origin could not take (down, busy, error) is given back. |
| Origin | A 4-thread CPU-only VM at home, behind a Cloudflare Tunnel | `sourcebound serve --public`: the real pipeline (hybrid retrieval, int8 ONNX openjev-flash rerank and check), at most two questions at once. |
| Language model | OpenRouter free tier | `nvidia/nemotron-3.5-super-vl:free`, falling back to `nvidia/nemotron-3-ultra-550b-a55b:free` and `google/gemma-4-26b-a4b-it:free`. |

Your question and the retrieved passages go to those providers, which may log them. When the limits are hit or the
origin is down, the page shows saved answers (`web/public/samples.json`), recorded in advance through the same
pipeline and models; they are labelled as saved. Papers and questions are from
[Qasper](https://huggingface.co/datasets/allenai/qasper) (CC BY 4.0). The Worker has its own tests with a mocked
origin: `cd web && npm test` (limits, secret forwarding, the fallback, oversized questions, the bot check) and
`npm run build`.

## Development

```bash
uv sync --extra rerank --extra eval
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 uv run pytest   # 136 tests, no network, no model downloads
```

Source layout: `src/sourcebound/` (`config`, `llm`, `embeddings`, `ingest/`, `store`, `graph`, `retrieval`,
`rerank`, `answer`, `agent`, `citations`, `api`, `cli`, `static/index.html`), `tests/`, `eval/`.

MIT licensed, see [LICENSE](LICENSE).
