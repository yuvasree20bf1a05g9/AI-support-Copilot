# 💬 AI Support Copilot

A retrieval-augmented (RAG) assistant for customer support teams. It answers
policy questions grounded in real support documents, cites its sources, and
honestly says **"I don't have enough information"** — with a one-click human
handoff — instead of guessing when it isn't confident.

Built after noticing, while working inside a high-volume customer support
operation, how much time gets spent manually searching policy documents to
compose an accurate, policy-aligned reply. This automates that lookup step.

**Stack:** Python · LangChain · ChromaDB · Sentence-Transformers · Groq (Llama 3.1) · Streamlit

---

## Demo

| Landing page | Grounded answer + citation |
|---|---|
| ![Landing page](docs/screenshots/01_landing_page.png) | ![Refund question with citation](docs/screenshots/02_refund_question_with_citation.png) |

| Retrieval across a second doc | Low-confidence → human handoff |
|---|---|
| ![Delivery partner question](docs/screenshots/03_delivery_partner_question.png) | ![Low confidence handoff](docs/screenshots/04_low_confidence_human_handoff.png) |

🎥 [Watch the full walkthrough](docs/demo_walkthrough.mp4)

> Drop the four screenshots and the mp4 (provided alongside this README) into
> a `docs/screenshots/` folder in your repo root so the links above resolve
> on GitHub.

---

## Architecture

```
                 ┌─────────────────┐
  Support docs → │   Ingestion      │
  (PDF / MD)     │  chunk + embed   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Chroma Vector   │
                 │       DB         │
                 └────────┬─────────┘
                          │  similarity search (top-k)
                          ▼
  User question → ┌─────────────────┐
                   │ Confidence gate  │──low──► "I don't know" + Human Handoff
                   └────────┬─────────┘
                          │ high
                          ▼
                 ┌─────────────────┐
                 │   LLM (Groq)     │
                 │  answer + cite   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  Streamlit Chat  │
                 │  UI + citations  │
                 └─────────────────┘
```

## Why these design choices

- **Confidence gate before generation, not after.** The pipeline checks
  retrieval similarity *before* calling the LLM. If the best-matching chunk
  isn't similar enough to the question, it skips generation entirely and
  routes to a human — this is what stops the copilot from confidently
  answering questions the knowledge base doesn't actually cover.
- **Cosine similarity, explicitly configured.** Chroma's default distance
  metric is raw L2, which is not bounded to `[0, 1]` and cannot be used
  directly as a "confidence" score. This project explicitly normalizes
  embeddings and configures the collection for cosine distance so the
  `1 - distance` confidence score is mathematically meaningful.
- **Citations carry chunk IDs, not just filenames**, so an agent (or a
  reviewer) can trace an answer back to the exact passage it came from.
- **Local, free embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) —
  no API cost or key needed for the retrieval half of the pipeline.

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd support-copilot
pip install -r requirements.txt
```

### 2. Get a free LLM API key

This project uses [Groq](https://console.groq.com) for generation — it has a
free tier and is fast. Sign up, create an API key, then:

```bash
cp .env.example .env
# edit .env and paste your key into GROQ_API_KEY=
```

You can run the app without a key too — it will show the retrieved, cited
context instead of an LLM-composed answer, so the retrieval half is fully
demonstrable even with zero API setup.

### 3. Build the vector store

```bash
python src/ingest.py
```

This indexes everything in `sample_docs/`. To use your own documents, drop
`.pdf`, `.md`, or `.txt` files into that folder first (or point
`SAMPLE_DOCS_DIR` in `src/config.py` at a different folder).

### 4. Run the app

```bash
streamlit run app.py
```

## Tuning the confidence threshold

`CONFIDENCE_THRESHOLD` in `src/config.py` controls how strict the "I don't
know" gate is. The value shipped here (`0.35`) is a starting point, not a
universal constant — the right value depends on your embedding model and
document set. To tune it for your own documents:

1. Ask 10-15 questions you *know* are covered by your docs, and note the
   `top_similarity` score the sidebar/expander shows for each.
2. Ask 10-15 questions you know are **not** covered, and note those scores.
3. Set the threshold roughly halfway between the lowest "known-good" score
   and the highest "known-bad" score.

## Project structure

```
support-copilot/
├── app.py                 # Streamlit chat UI
├── src/
│   ├── config.py           # all tunable settings in one place
│   ├── ingest.py            # load → chunk → embed → store
│   └── rag_pipeline.py      # retrieve → confidence gate → generate → cite
├── sample_docs/            # example support policy documents
├── docs/                   # screenshots + demo video for this README
├── requirements.txt
└── .env.example
```

## What this project deliberately does NOT include

This was scoped to demonstrate the core RAG engineering competently rather
than claim enterprise breadth it doesn't have. Left out on purpose:
multi-channel deployment (WhatsApp/Slack/Teams/voice IVR), PII redaction,
jailbreak resistance testing, SOC 2 / GDPR compliance tooling, on-prem
deployment, multi-model routing, and full observability. Those are real,
valuable engineering problems — just not what this project is claiming to
solve.
