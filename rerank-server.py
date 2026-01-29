import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://localhost:30000/v1")
SGLANG_API_KEY = os.getenv("SGLANG_API_KEY", "EMPTY")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-8B")

client = OpenAI(
    api_key=SGLANG_API_KEY,
    base_url=SGLANG_BASE_URL,
)

# ---------- Request / Response Schemas ----------

class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: List[str]
    top_n: Optional[int] = None  # if None, return all

class RerankedDoc(BaseModel):
    document: str
    index: int
    score: float

class RerankResponse(BaseModel):
    object: str = "list"
    data: List[RerankedDoc]

app = FastAPI()


def build_rerank_prompt(query: str, docs: List[str]) -> str:
    # Extremely simple text-only template; customize as needed.
    numbered_docs = "\n".join(
        f"Document {i}: {doc}" for i, doc in enumerate(docs)
    )
    return (
        "You are a relevance ranking model.\n"
        "Given the user query and the numbered documents, "
        "return a JSON list of objects, each with keys: index, score.\n"
        "Score should be a real number where higher means more relevant.\n\n"
        f"Query: {query}\n\n"
        f"{numbered_docs}\n\n"
        "Respond with JSON ONLY, no explanation."
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    model_name = request.model or RERANK_MODEL
    prompt = build_rerank_prompt(request.query, request.documents)

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=256,
    )

    content = completion.choices[0].message.content.strip()

    import json
    raw = json.loads(content)

    scores = []
    for item in raw:
        idx = int(item["index"])
        score = float(item["score"])
        scores.append((idx, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    if request.top_n is not None:
        scores = scores[: request.top_n]

    result_docs = [
        RerankedDoc(
            document=request.documents[idx],
            index=idx,
            score=score,
        )
        for idx, score in scores
    ]

    return RerankResponse(data=result_docs)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8102)

