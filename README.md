# AI Support Ticket Analyzer

> An AI-powered customer support ticket analytics system for DOTMappers IT Pvt. Ltd.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 1. Overview

This system ingests a structured CSV of 500 customer support tickets and makes them
queryable through natural language. A user can ask questions like
*"Which agent has the lowest average rating?"* and receive an accurate, deterministic answer.

**The key architectural principle:** the LLM is used **only** for natural-language
understanding (intent extraction). All numerical calculations are performed
deterministically in Python using Pandas — never by the LLM.

---

## 2. Features

- **Natural-language queries** over structured ticket data
- **11+ query types**: count, average, min/max, list, group-by, ranking, date filtering
- **4 anomaly detection rules**: statistical + rule-based, no LLM involvement
- **REST API** with FastAPI — fully documented at `/docs`
- **Professional dashboard UI** — KPIs, AI query, anomaly table
- **LLM provider abstraction** — Groq (free), Ollama (local), OpenAI-compatible
- **Comprehensive test suite** — 40+ tests, LLM mocked for reliability
- **Zero cost to run** — Groq free tier or Ollama locally

---

## 3. Architecture

```
User Question
      │
      ▼
  FastAPI /api/query
      │
      ▼
  LLM Provider (Groq / Ollama / OpenAI)
  [Intent extraction ONLY — no calculations]
      │
      ▼
  StructuredQuery (Pydantic validation)
  [Allowlisted intents, fields, operations]
      │
      ▼
  Deterministic Query Engine
  [Pure Python + Pandas]
      │
      ▼
  support_tickets.csv (in-memory DataFrame)
      │
      ▼
  Structured Response → Natural Language Answer → UI
```

**Anomaly Detection (separate pipeline):**
```
support_tickets.csv → AnomalyDetector → 4 rule-based checks → AnomalyRecord list
```

### Component Map

| Component | File | Responsibility |
|-----------|------|----------------|
| Data Service | `app/services/data_service.py` | Load, validate, normalise CSV |
| LLM Service | `app/services/llm_service.py` | Provider abstraction, intent parsing |
| Query Engine | `app/services/query_engine.py` | Deterministic execution |
| Anomaly Detector | `app/services/anomaly_detector.py` | Rule-based anomaly detection |
| Schemas | `app/models/schemas.py` | Pydantic models, allowlists |
| Prompts | `app/prompts/query_prompt.py` | Versioned LLM system prompt |
| API | `app/api/` | FastAPI route handlers |
| UI | `static/` | Vanilla HTML/CSS/JS dashboard |

---

## 4. Technology Stack

| Technology | Version | Rationale |
|------------|---------|-----------|
| **Python** | 3.11+ | Type hints, performance, ecosystem |
| **FastAPI** | 0.104 | Async, auto-docs, Pydantic integration |
| **Pydantic** | v2 | Schema validation with allowlists |
| **Pandas** | 2.2 | Efficient in-memory analytics (sufficient for 500 rows) |
| **Groq** | 0.4 | Free-tier LLM, fast inference (llama3-8b-8192) |
| **Uvicorn** | 0.24 | ASGI server |
| **pytest** | 7.4 | Testing with LLM mocking |
| **Vanilla JS** | — | No framework overhead for a simple dashboard |

---

## 5. Architecture Decisions

### Why the LLM is ONLY used for intent extraction
The LLM converts natural language → structured JSON (intent + filters). Python then
executes the query deterministically. This means:
- Answers are always accurate (no hallucination in calculations)
- Results are reproducible
- The LLM can be swapped without changing query logic
- Failures in the LLM return graceful errors, not wrong answers

### Why no vector database / RAG?
The data is structured (CSV with known columns). Semantic search over ticket content
is not required for analytics queries. Pandas is faster and deterministic.

### Why Pandas instead of a database?
500 rows fit comfortably in memory. Loading once at startup gives sub-millisecond
query times. For larger datasets, swapping to PostgreSQL + SQLAlchemy would require
changes only to `data_service.py` and `query_engine.py`.

### Why Groq (free tier) instead of OpenAI?
Groq's free tier provides ~14,400 requests/day with llama3-8b-8192 — sufficient
for any demo or evaluation. No credit card required.

### Why no LangChain?
LangChain adds complexity without benefit here. The prompt → JSON → validation →
engine pipeline is straightforward and better implemented directly.

### How it scales
- Replace `DataService` with a database-backed service
- Add Redis caching for anomaly detection results
- Deploy behind a load balancer (FastAPI is async-ready)
- Add authentication middleware

---

## 6. Setup

### Prerequisites
- Python 3.11+
- pip

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment
```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## 7. LLM Setup

### Option A: Groq (Recommended — Free)
1. Sign up at https://console.groq.com
2. Create an API key
3. Add to `.env`:
   ```bash
   LLM_PROVIDER=groq
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

### Option B: Ollama (Local — Fully Offline)
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2
```
Then in `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### Option C: OpenAI-compatible
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo
```

---

## 8. Running

```bash
# Start the application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or using the entry point
python app/main.py
```

Open: http://localhost:8000

- **Dashboard UI**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 9. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Application health check |
| `GET` | `/api/stats` | Dashboard KPIs and statistics |
| `GET` | `/api/anomalies` | All detected anomalies |
| `GET` | `/api/anomalies?severity=HIGH` | Filter anomalies by severity |
| `GET` | `/api/anomalies?anomaly_type=LOW_CUSTOMER_RATING` | Filter by type |
| `POST` | `/api/query` | Natural-language query |

### POST /api/query

**Request:**
```json
{"question": "How many critical tickets are unresolved?"}
```

**Response:**
```json
{
  "question": "How many critical tickets are unresolved?",
  "intent": "COUNT_TICKETS",
  "answer": "There are 31 tickets that are unresolved (Open or Escalated) and with priority 'Critical'.",
  "data": {"count": 31},
  "structured_query": {
    "intent": "COUNT_TICKETS",
    "filters": {
      "status": "Unresolved",
      "priority": "Critical"
    }
  }
}
```

---

## 10. Example Questions & Actual Outputs

All values below were deterministically computed from the provided `support_tickets.csv` (500 records):

| Natural Language Question | Detected Intent | Deterministic Answer from Dataset |
|---------------------------|-----------------|-----------------------------------|
| *How many tickets are currently open?* | `COUNT_TICKETS` | **There are 111 tickets with status 'Open'.** |
| *How many critical tickets are unresolved?* | `COUNT_TICKETS` | **There are 31 tickets that are unresolved (Open or Escalated) and with priority 'Critical'.** |
| *Which agent resolved the most tickets?* | `TOP_AGENT_RESOLVED` | **Agent AGT-09 resolved the most tickets with 37 resolved tickets.** |
| *Which agent has the lowest average customer rating?* | `TOP_AGENT_RATING` | **Agent AGT-08 has the lowest average customer rating of 3.48/5.0.** |
| *What is the average customer rating for Technical category tickets?* | `AVERAGE_METRIC` | **The average customer rating in category 'Technical' is 3.74.** |
| *What is the average response time for High priority tickets?* | `AVERAGE_METRIC` | **The average response time (hours) with priority 'High' is 2.51.** |
| *Which category has the most tickets?* | `TOP_CATEGORY` | **The "General" category has the most tickets with 189.** |
| *How many Billing tickets are unresolved?* | `COUNT_TICKETS` | **There are 58 tickets that are unresolved (Open or Escalated) and in category 'Billing'.** |
| *Show me all Critical tickets not resolved within 12 hours.* | `LIST_TICKETS` | **Found 34 tickets with priority 'Critical' and not resolved within 12.0 hours.** (Matches critical tickets unresolved or resolved in > 12h) |

### Real Dataset KPI Summary
- **Total Tickets**: 500 (Open: 111, Resolved: 327, Escalated: 62)
- **Critical Tickets**: 55 (Unresolved Critical: 31, Resolved Critical: 24)
- **Avg First Response Time**: 2.62 hours
- **Avg Resolution Time**: 19.16 hours
- **Avg Customer Rating**: 3.75 / 5.0
- **Detected Anomalies**: 144 (Critical: 31, High: 70, Medium: 43)

---

## 11. Anomaly Detection

Four deterministic rules (no LLM involvement):

### Rule 1 — LONG_RESOLUTION_TIME
- **Method:** Statistical (mean + 2×std of resolved tickets' resolution time)
- **Severity:** HIGH if > mean + 3×std, MEDIUM otherwise
- **Why not hardcoded?** The threshold adapts to actual data distribution

### Rule 2 — HIGH_PRIORITY_OVERDUE
- **Condition:** Open/Escalated ticket with priority High or Critical, older than 24 hours
- **Reference date:** End of dataset range (2024-03-30)
- **Severity:** CRITICAL for Critical priority, HIGH for High priority

### Rule 3 — SLOW_FIRST_RESPONSE
- **Method:** Statistical (mean + 2×std of response_time_hrs across all tickets)
- **Severity:** MEDIUM

### Rule 4 — LOW_CUSTOMER_RATING
- **Condition:** Resolved ticket with customer_rating ≤ 2.0
- **Severity:** HIGH for rating = 1, MEDIUM for rating = 2

---

## 12. Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_data_service.py -v
pytest tests/test_query_engine.py -v
pytest tests/test_anomaly_detector.py -v
pytest tests/test_api.py -v

# With coverage report
pytest tests/ --tb=short
```

**Note:** All tests mock the LLM provider — no API key required to run tests.

---

## 13. Dataset Schema

Internal canonical column names (after normalisation):

| Column | Type | Description |
|--------|------|-------------|
| `ticket_id` | string | e.g. TKT-001 |
| `created_at` | datetime | 2024-01-01 to 2024-03-30 |
| `category` | string | General, Billing, Technical |
| `priority` | string | Low, Medium, High, Critical |
| `status` | string | Open, Resolved, Escalated |
| `response_time_hrs` | float | Hours until first response |
| `resolution_time_hrs` | float | Hours until resolved (NaN if unresolved) |
| `agent_id` | string | AGT-01 through AGT-12 |
| `customer_rating` | float | 1.0–5.0 (NaN if unresolved) |
| `issue_summary` | string | Short description |

**Column alias normalisation:** The data service automatically maps `resp_time_hrs`,
`resol_time_hrs`, `cust_rating` etc. to their canonical names.

### Date filter temporal context
The dataset covers 2024-01-01 to 2024-03-30. Relative date expressions like
"this week" are resolved against the **dataset's end date** (2024-03-30), not
the current wall-clock date. This ensures historical datasets produce meaningful
results. The behaviour is documented in `app/utils/helpers.py`.

---

## 14. Known Limitations

1. **LLM JSON adherence:** Very small/quantized models may occasionally return
   malformed JSON. The system catches and reports this gracefully.
2. **500-row dataset:** Optimised for in-memory Pandas. Not suitable for millions
   of rows without adding a database.
3. **Date interpretation:** "this week" / "this month" uses dataset reference date.
   If real-time ticket ingestion is added, use wall-clock time.
4. **No authentication:** Designed for internal analytics. Add OAuth2/JWT for
   production use.
5. **Single-node:** No horizontal scaling as-is. Add Redis caching + load balancer
   for production.

---

## 15. Future Improvements

- **PostgreSQL + SQLAlchemy** for larger datasets
- **Background data ingestion** with file watchers or webhook receivers
- **Redis caching** for anomaly results (recalculate on schedule)
- **Authentication** — OAuth2 / API key middleware
- **Observability** — OpenTelemetry tracing, structured logging to ELK
- **More anomaly rules** — e.g. sudden spike in ticket volume, agent overload
- **Streaming responses** — Server-sent events for long-running queries
- **Multi-tenant** — Separate datasets per team/organisation
- **Export** — CSV/PDF report generation
