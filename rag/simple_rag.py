from .loader import load_pdf
from .embeddings import Embedder
from chatbot.bot import Chatbot
import numpy as np


class SimpleRAG:
    def __init__(self, llm_api_key=None, embedder_model="all-MiniLM-L6-v2"):
        self.embedder = Embedder(embedder_model)
        self.chatbot = Chatbot(api_key=llm_api_key)

        self.cv_text = None
        self.cv_vector = None

        # similarity cutoff
        self.threshold = 0.1

    # PDF ingestion
    def ingest_pdf(self, path: str):
        """Load the entire CV and embed it once."""
        self.cv_text = load_pdf(path)
        self.cv_vector = self.embedder.embed([self.cv_text])[0]
        return "CV loaded successfully!"

    # Answering user questions
    def answer(self, question: str):
        # Embed question
        question_vec = self.embedder.embed([question])[0]

        # Compute cosine similarity with CV
        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_cv = cosine(question_vec, self.cv_vector)

        # handle irrelevant queries
        if sim_cv < self.threshold:
            return {
                "answer": "I don't know.",
                "relevance": round(sim_cv, 3)
            }

        # Build LLM prompt (no conversation history)
        prompt = (
            "You are an expert technical recruiter and CV analyst. Your job is to evaluate the "
            "candidate strictly based on the provided CV.\n\n"
            "RULES:\n"
            "1. Use ONLY information found in the CV content.\n"
            "2. If the CV does not explicitly mention something, reply with: "
            "'I don't know, there is no information regarding that in the CV.'\n"
            "3. When giving opinions or assessments, ensure they are directly justified by the CV.\n"
            "4. Do NOT infer, assume, guess, or hallucinate anything that is not written.\n"
            "5. Keep answers from a recruiter's perspective.\n"
            "6. Keep answers short and concise.\n\n"
            f"CV Content:\n{self.cv_text}\n\n"
            f"Question: {question}\n"
            "Answer as a recruiter:"
        )

        # LLM call
        response = self.chatbot.get_response(prompt)

        return {
            "answer": response,
            "relevance": round(sim_cv, 3)
        }

