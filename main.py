from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag.simple_rag import SimpleRAG
from rag.context_aware_rag import ContextAwareRAG
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_KEY= os.getenv("GROQ_API_KEY")

app = FastAPI()

#Simple RAG
agent= SimpleRAG(llm_api_key=GROQ_KEY)

#context aware RAG
context_agent= ContextAwareRAG(llm_api_key=GROQ_KEY)

class QuestionRequest(BaseModel):
    question: str
    
@app.on_event("startup")
def startup():
    print("Info Loading CV...")
    agent.ingest_pdf("cv.pdf")
    context_agent.ingest_pdf("cv.pdf")   # <--- ADD THIS LINE
    print("Ready for queries.")
    
@app.post("/ask_srag")
def ask_srag(re: QuestionRequest):    
    if not re.question.strip():
        raise HTTPException(status_code=400,detail="Question cannot be empty.")
    
    return agent.answer(re.question)

@app.post("/ask_cwrag")
def ask_cwrag(re: QuestionRequest):    
    if not re.question.strip():
        raise HTTPException(status_code=400,detail="Question cannot be empty.")
    
    return context_agent.answer(re.question)