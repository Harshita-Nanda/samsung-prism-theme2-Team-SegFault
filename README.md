\# Smart Guided Troubleshooting Engine



Samsung PRISM GenAI Hackathon | Theme 2



An API service that turns vague customer complaints (e.g. "my phone is super slow and hot") into clean, validated, machine-actionable troubleshooting plans, delivered as strict JSON.



\## System Architecture



User Complaint

&#x20;   |

&#x20;   v

1\. Query Enrichment (Gemini via Google GenAI SDK)

&#x20;  Normalizes colloquial text into a canonical technical query and generates cache-key paraphrases

&#x20;   |

&#x20;   v

2\. Pydantic JSON Extraction

&#x20;  Extracts Goal, categorized Actions and UI Steps; output is validated against a strict Pydantic schema (no web URLs allowed)

&#x20;   |

&#x20;   v

3\. Deeplink Mapping and Action Ordering (BM25 + dense embeddings)

&#x20;  Maps each extracted UI step to an exact Settings deeplink from a \~575-entry catalog (dual retrieval) and orders actions safely

&#x20;   |

&#x20;   v

4\. Fast-Path Semantic Cache

&#x20;  Repeat or similar queries are served from the cache for sub-300ms responses

&#x20;   |

&#x20;   v

JSON response (FastAPI)



Stack: Python 3.11, FastAPI, Uvicorn, Pydantic, Google GenAI SDK, sentence-transformers, rank-bm25, Docker.



\## API Endpoints



\### GET /health

Liveness check.



curl http://localhost:8000/health



\### POST /resolve

Accepts a natural-language complaint and returns a structured troubleshooting plan.



Request body:

{

&#x20; "complaint": "phone battery drains fast",

&#x20; "siis\_response": "optional reference article text used to ground the answer"

}



Response body:

{

&#x20; "answer": "structured JSON plan with query, contexts, deeplinks",

&#x20; "cache\_hit": false,

&#x20; "latency\_ms": 850.2,

&#x20; "cost\_usd": 0.01

}



Example:

curl -X POST http://localhost:8000/resolve -H "Content-Type: application/json" -d "{\\"complaint\\": \\"phone battery drains fast\\"}"



Interactive docs are available at http://localhost:8000/docs



\## Running with Docker



Prerequisites: Docker Desktop installed, plus a Gemini API key.



1\. Clone the repository

git clone https://github.com/Harshita-Nanda/samsung-prism-theme2-Team-SegFault.git

cd samsung-prism-theme2-Team-SegFault



2\. Provide your API key (kept out of git)

cp .env.example .env

edit .env and set GEMINI\_API\_KEY=your key



3\. Build and start (single command)

docker compose up --build



The service is now live at http://localhost:8000



Run in the background and stop:

docker compose up --build -d

docker compose down



Without Compose:

docker build -t prism-troubleshoot .

docker run --rm -p 8000:8000 -e GEMINI\_API\_KEY=%GEMINI\_API\_KEY% prism-troubleshoot



\## Project Structure



main.py - FastAPI app (endpoints, cache wiring)

pipeline.py - Orchestrates Phase 1 -> Phase 2

cache.py - Semantic fast-path cache

engine/ - Phase 1: query enrichment and structure extraction

catalog.py - Phase 2: deeplink catalog (BM25 + dense retrieval)

mapper.py - Phase 2: maps steps to deeplinks and orders actions

schema.py - Shared Pydantic schema (Goal, etc.)

deeplinks.json - \~575-entry masked deeplink catalog

requirements.txt

Dockerfile

docker-compose.yml

.env.example



\## Submission



Release tag: PRISM\_GENAI\_HACKATHON\_Y2026

Demo video: [Google Drive Walkthrough Video](https://drive.google.com/file/d/1UQJK8GgxAuzQIZbgipz040OZmuzmU_YO/view?usp=sharing)

Presentation: MSRIT_SegFault



\## Team



Person 1 - LLM and Prompt Engineering (Phase 1)

Person 2 - Backend API and Caching (Phase 3 and 4)

Person 3 - Search, Retrieval and Deeplinks (Phase 2)

Person 4 - QA, DevOps and Submission

