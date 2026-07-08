# Project Guide: Agentic Research Assistant

> Written for someone joining this project for the first time. If you're new to Python backends, AI agents, or React, this is meant to be read top to bottom with no prior context.

---

## 1. What is this project?

It's a tool that reads a stack of research papers and answers questions about them, **with proof for every claim it makes**.

You give it a question like *"What methods do these papers use for grounding, and what gaps remain?"* and it comes back with an answer where every sentence has a little `[1]`, `[2]`, `[3]` marker next to it — a footnote pointing at the exact passage in the exact paper that supports that sentence. Alongside the answer, it also pulls out three structured lists: the **methods** used, the **key findings**, and the **research gaps** the papers leave open.

There are two halves to the codebase:

1. **The backend** (Python) — does the actual thinking. Reads papers, finds relevant passages, asks Claude (Anthropic's AI model) to draft an answer, checks that answer against the sources, and loops back for more evidence if the answer is weak.
2. **The frontend** (React, in `frontend/`) — a web page where you type a question and watch the answer come together, with clickable citations and a little animated "agent rail" showing which stage (retrieve / draft / check) is currently running.

## 2. Why does this exist? What problem does it solve?

If you dump a pile of PDFs into a single ChatGPT-style prompt, two things go wrong:

- **It makes things up.** With no way to check its work, the model will confidently state numbers or claims that aren't actually in the source material.
- **It skips things.** A single pass over a big pile of text tends to gloss over the parts that don't fit neatly into a short answer.

This project's answer to that is to **not use a single prompt**. Instead, three separate AI "agents" each do one narrow job and hand off to the next, and there's a built-in self-check that can send the work back for another round:

| Agent | Job |
|---|---|
| **Retriever** | Given the current question, find the most relevant chunks of text across all the indexed papers. |
| **Summarizer** | Given those chunks, write a structured answer where every factual sentence carries a `[n]` marker pointing at a specific chunk. |
| **Evaluator** | A second, independent look at the summarizer's answer: is every claim actually backed by its citation? Is the question fully answered? If not, it writes a better search query and the whole thing loops back to the retriever. |

This retrieve → summarize → evaluate loop is what "agentic" means here — the program's own critique of its work decides whether to stop or do another round, up to a safety cap (`max_iterations`, default 3) so it can't loop forever.

## 3. The technologies involved (and what each one is for)

If any of these are unfamiliar, here's the one-line version of what each does in this project:

- **Python** — the language the backend is written in.
- **[FastAPI](https://fastapi.tiangolo.com/)** — a Python web framework. It's what turns the research pipeline into an HTTP API (`POST /research`) that anything (a browser, `curl`, another program) can call over the network.
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — a library for building "graphs" of steps where the next step isn't always fixed — it can loop back based on a decision. This is what implements the retrieve → summarize → evaluate → (loop or stop) flow.
- **[Anthropic's Claude API](https://docs.anthropic.com/)** — the actual AI model doing the reading, writing, and judging. The summarizer and evaluator are each a call to Claude.
- **[FAISS](https://github.com/facebookresearch/faiss)** (Facebook AI Similarity Search) — a library for very fast "find the most similar text" search. Every chunk of every paper is turned into a vector (a list of numbers representing its meaning); FAISS finds the chunks whose vectors are closest to the question's vector. This is what "retrieval" means in RAG (Retrieval-Augmented Generation).
- **[sentence-transformers](https://www.sbert.net/)** — turns text into those vectors (called "embeddings"). There's also a dependency-free "hashing" fallback used in offline/test mode so you don't need to download a model just to try things out.
- **[Pydantic](https://docs.pydantic.dev/)** — validates data shapes. Every AI response gets parsed into a strict typed object (`schemas.py`) so a malformed model response fails loudly instead of silently corrupting the output.
- **React + [Vite](https://vitejs.dev/)** — the frontend. React is the UI library (components, state); Vite is the build tool/dev server that makes `npm run dev` fast.
- **[lucide-react](https://lucide.dev/)** — the icon set used in the UI (search icon, book icon, shield icon, etc.).
- **pytest** — the Python testing framework used for `tests/`.
- **Docker / Render** — how the backend gets deployed to the internet (see §8).

You do **not** need an Anthropic API key to run or explore this project. Every piece has an offline fallback (a "stub" that fakes plausible output using the real retrieved text, and a hashing embedder instead of a downloaded model), so the entire loop — real retrieval, real graph, real citation numbering — runs with no network calls at all. This is called **dry-run / offline mode** throughout the code, and it's the default.

## 4. How the pieces fit together (architecture)

```
                          +----------------------- retrieve <-------------------+
                          |                           |                        |
    question ---(1)-------+                           v                        | (5) needs more evidence,
                                                   summarize                    |     refined query,
                                                       |                        |     budget left
                                                       v                        |
                                                   evaluate ---------------------+
                                                       |
                                                       | (4) sufficient, or budget spent
                                                       v
                                                   compose ----> final answer + citations
```

1. **Ingest** — papers (PDF/text/Markdown, or fetched from arXiv) get split into overlapping chunks (`ingestion/chunking.py`), each chunk gets embedded into a vector, and all vectors go into a FAISS index (`rag/vector_store.py`).
2. **Retrieve** (`agents/retriever.py`) — for the current query, FAISS returns the top-k most similar chunks. On later loop iterations, new chunks are merged with what was already found.
3. **Summarize** (`agents/summarizer.py`) — the retrieved chunks are numbered and handed to Claude with instructions: answer using only these sources, and tag every factual sentence with `[n]`. The JSON that comes back is validated into a `PaperSummary` object.
4. **Evaluate** (`agents/evaluator.py`) — a second, separate Claude call scores whether the answer is grounded (every claim supported) and how much of the question is covered (`coverage_score`, 0–1), then decides `sufficient` or `needs_more`.
5. **Loop or stop** — if `needs_more` and there's iteration budget left, the evaluator's suggested `refined_query` goes back to step 2. Otherwise:
6. **Compose** (`report.py`) — everything is rendered into a final Markdown report: the answer, the methods/findings/gaps lists, the evaluation verdict, and a numbered reference list matching every `[n]` back to its source.

All of this logic is orchestrated as a **LangGraph state machine** in `graph/workflow.py` — think of it as a flowchart where each box is a Python function and the arrows (including the loop-back arrow) are decided by code, not hardcoded.

### Where things live

```
agentic-research-assistant/
├── src/research_assistant/
│   ├── config.py          # All settings, loaded from environment variables / .env
│   ├── schemas.py         # The typed data models (Chunk, PaperSummary, Evaluation, ResearchResult, ...)
│   ├── llm.py             # The Claude API client (with retries) + the offline stub version
│   ├── context.py         # Numbers the sources and keeps [n] citations lined up correctly
│   ├── report.py          # Turns the final result into a Markdown report
│   ├── factory.py         # Wires everything together (which embedder, which client, etc.)
│   ├── cli.py             # The `research-assistant` command-line tool
│   ├── webapp.py          # The FastAPI web service (this is what the frontend talks to)
│   ├── ingestion/
│   │   ├── loaders.py     # Reads PDF / text / Markdown files, or pulls from arXiv
│   │   └── chunking.py    # Splits long text into overlapping chunks
│   ├── rag/
│   │   ├── embeddings.py  # Turns text into vectors (sentence-transformers or hashing)
│   │   └── vector_store.py# The FAISS index wrapper (add/search/save/load)
│   ├── agents/
│   │   ├── retriever.py
│   │   ├── summarizer.py
│   │   └── evaluator.py
│   └── graph/
│       ├── state.py       # The shape of data passed between graph nodes
│       └── workflow.py    # The graph itself: retrieve → summarize → evaluate → loop/stop
├── tests/                 # pytest tests for every piece above
├── examples/sample_papers/# 3 synthetic example papers used by the demo
├── frontend/               # The React + Vite web UI
│   └── src/App.jsx         # The entire UI in one file (styles included)
├── data/index_demo/        # A pre-built FAISS index over the sample papers (for the deployed demo)
├── Dockerfile, render.yaml # Deployment config
├── Makefile                # Shortcut commands (see below)
└── .env.example             # Template for your local secrets/config
```

## 5. Setting up your machine (tools you need)

You'll need two completely separate toolchains, because this is two programs (a Python API and a JavaScript web app) that just happen to live in one repo.

### For the backend (Python)
- **Python 3.9+** (check with `python3 --version`)
- `pip` (comes with Python)

### For the frontend (React)
- **Node.js 18+** and **npm** (comes bundled with Node). Check with `node -v` and `npm -v`. If you don't have Node, install it from [nodejs.org](https://nodejs.org/) (the LTS version).

### Optional
- An **Anthropic API key** (from [console.anthropic.com](https://console.anthropic.com/)) — only needed if you want real Claude-generated answers instead of the offline stub. Not required to explore the project.

## 6. Running the backend, step by step

All commands below assume you're in the project's root folder in your terminal.

**Step 1 — create and activate a virtual environment.** A virtual environment is an isolated folder of Python packages just for this project, so it doesn't clash with anything else on your machine.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`. You'll need to run that `source` command again every time you open a new terminal tab and want to work on this project (once activated, it stays active for that terminal session only).

**Step 2 — install the project.**

```bash
pip install -e ".[dev]"
```

`-e` means "editable" — it links the package to this source folder instead of copying it, so your code edits take effect immediately without reinstalling. `[dev]` also pulls in `pytest` and `ruff` (the linter).

**Step 3 — try the offline demo (no API key, no network).**

```bash
python -m research_assistant demo
```

This ingests the three sample papers in `examples/sample_papers/`, runs the full retrieve → summarize → evaluate loop with a stub model that fakes plausible output from the real retrieved text, and prints a Markdown report to your terminal. This proves the whole pipeline works end to end before you touch any API keys.

**Step 4 — run the actual web server.**

```bash
uvicorn research_assistant.webapp:app --reload
```

- `uvicorn` is the program that actually runs a FastAPI app and listens for HTTP requests.
- `research_assistant.webapp:app` means "the variable named `app` inside `research_assistant/webapp.py`".
- `--reload` restarts the server automatically whenever you edit the code — very useful while developing.

Once it's running, you'll see log lines ending in something like `startup complete: mode=dry_run chunks=25`, and the server is listening on `http://127.0.0.1:8000`. Leave this terminal running — it's now your local backend.

Check it's alive from another terminal (or your browser):
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","mode":"dry_run"}
```

Or open `http://127.0.0.1:8000/docs` in a browser — FastAPI auto-generates an interactive page where you can try the `/research` endpoint directly, no frontend needed.

**Step 5 (optional) — use the real Claude API instead of the stub.**

```bash
cp .env.example .env
```
Open `.env` in an editor and set:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```
Restart `uvicorn` (Ctrl+C, then run the command again) — it reads `.env` on startup, and `webapp.py` will automatically switch to `mode=live` because it now sees a key.

**Step 6 (optional) — the command-line tool, for real usage beyond the demo.**

```bash
# Index your own papers (folder of PDFs/text/Markdown)
python -m research_assistant ingest path/to/your/papers --dry-run

# Ask a question against that index
python -m research_assistant research "What are the main limitations?" --dry-run
```
Drop `--dry-run` once you've set up your API key to get real Claude answers instead of the stub.

## 7. Running the frontend, step by step

Open a **second terminal tab** (leave the backend running in the first one).

**Step 1 — go into the frontend folder and install its dependencies.**

```bash
cd frontend
npm install
```

This reads `frontend/package.json` and downloads React, Vite, and lucide-react into a `frontend/node_modules/` folder (this folder is intentionally not committed to git — see `frontend/.gitignore` — everyone regenerates it locally with `npm install`).

**Step 2 — start the dev server.**

```bash
npm run dev
```

Vite will print a local URL, typically `http://localhost:5173`. Open that in your browser.

**What you'll see by default:** the app runs entirely on **built-in demo data** — no backend call at all. This is intentional, so the UI itself is testable in isolation. You'll notice the little pill in the top-right corner says **"offline demo"**.

**Step 3 (optional) — connect it to your local backend for real answers.**

```bash
cp .env.example .env
```
Edit `frontend/.env`:
```
VITE_API_URL=http://127.0.0.1:8000
VITE_API_KEY=
```
Stop and restart `npm run dev` (Vite only reads `.env` files at startup, not live). Now the header pill should say **"live api"**, and asking a question sends a real `POST /research` request to your local FastAPI server instead of using canned data.

> Why does this need restarting but the backend's `--reload` doesn't? `--reload` only watches Python source files. Environment variables are read once when a process starts, for both Vite and uvicorn — the difference you're seeing is `--reload` catching code changes, not env changes. If you edit `.env` for the backend, you'd need to restart `uvicorn` too.

## 8. How it's deployed (for context, not required to develop locally)

- **Backend**: `Dockerfile` packages the FastAPI app into a container; `render.yaml` is a "blueprint" that [Render](https://render.com/) reads to deploy it automatically on their free tier. The `/health` endpoint exists specifically so Render's load balancer can check the service is alive.
- **CORS**: because the frontend (running on one origin, e.g. `localhost:5173` or a deployed static site) needs to call the backend (running on a different origin, e.g. `onrender.com`), the backend has to explicitly allow that with CORS headers — see `CORS_ORIGINS` in `.env.example` and the `CORSMiddleware` setup in `webapp.py`. Without this, browsers block the request even though `curl` would work fine.
- **Frontend**: not yet wired to an auto-deploy — `npm run build` produces a static `frontend/dist/` folder that can be hosted anywhere that serves static files (Vercel, Netlify, GitHub Pages, etc.).

## 9. Running the tests

```bash
source .venv/bin/activate   # if not already active
pytest
```

This runs everything in `tests/`: the chunker, schema validation, citation numbering, the hashing embeddings, JSON extraction from model output, the FAISS store, the web API (including auth and CORS), and a full offline end-to-end run of the whole agent graph. Tests that need `faiss-cpu` or `langgraph` skip themselves automatically if those packages aren't installed, so even a minimal setup can run *something* — but `pip install -e ".[dev]"` from step 2 installs everything, so you should see all of them run.

There's no automated test suite for the frontend yet — verifying it means running `npm run dev` and clicking around.

## 10. Quick reference: commands you'll actually use day to day

```bash
# --- backend ---
source .venv/bin/activate
uvicorn research_assistant.webapp:app --reload   # run the API
pytest                                           # run backend tests
python -m research_assistant demo                # quick offline sanity check

# --- frontend ---
cd frontend
npm run dev                                      # run the UI

# --- Makefile shortcuts (from the project root) ---
make demo             # same as `python -m research_assistant demo`
make test             # same as `pytest`
make frontend-dev     # same as `cd frontend && npm run dev`
```

## 11. Glossary (things you might not have heard before)

- **RAG (Retrieval-Augmented Generation)** — instead of asking an AI model to answer purely from what it memorized during training, you first retrieve relevant real text and hand that to it as context. Reduces made-up answers.
- **Embedding / vector** — a list of numbers that represents the *meaning* of a piece of text, produced by a model trained so that similar meanings end up as nearby numbers. This is what makes "search by meaning, not just keywords" possible.
- **Grounded** — a claim is "grounded" if it's actually backed up by the cited source, as opposed to being a plausible-sounding guess.
- **Dry-run / offline / stub** — running the whole system with fake-but-realistic stand-ins for the parts that would otherwise need an API key or a model download, so you can verify the plumbing works for free.
- **CORS (Cross-Origin Resource Sharing)** — a browser security rule that blocks a web page from calling an API on a different domain unless that API explicitly says it's allowed to.
- **Agent** — in this project, "agent" just means "one focused step, backed by an LLM call, with a clear single job" (retrieve, summarize, or evaluate) — not anything more mysterious than that.
