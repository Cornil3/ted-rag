from fastapi import FastAPI
from pydantic import BaseModel
from api.rag import generate_answer
from api.config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
import uvicorn

app = FastAPI()


class PromptRequest(BaseModel):
    question: str


@app.get("/")
def health_check():
    return "I'M ALIVE!"


@app.post("/api/prompt")
def prompt(req: PromptRequest):
    answer = generate_answer(req.question)
    return answer


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8001, reload=True)
