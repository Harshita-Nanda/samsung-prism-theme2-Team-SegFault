from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from pipeline import run_pipeline
import time
import json
from cache import SemanticCache

app = FastAPI()

cache = SemanticCache(threshold=0.50)


class ResolveRequest(BaseModel):
    complaint: str
    siis_response: Optional[str] = None


class ResolveResponse(BaseModel):
    answer: str
    cache_hit: bool
    latency_ms: float
    cost_usd: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest):
    start = time.perf_counter()

    cached_answer, score = cache.get(req.complaint)

    if cached_answer:
        answer = cached_answer
        hit = True
    else:
        payload = run_pipeline(req.complaint, req.siis_response)
        answer = json.dumps(payload, ensure_ascii=False)
        cache.put(req.complaint, answer)
        hit = False

    latency = (time.perf_counter() - start) * 1000
    return ResolveResponse(
        answer=answer,
        cache_hit=hit,
        latency_ms=latency,
        cost_usd=0.0 if hit else 0.01,
    )