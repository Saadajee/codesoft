#rag/context_aware_rag.py

from .loader import load_pdf
from .embeddings import Embedder
from chatbot.bot import Chatbot
import numpy as np

class ContextAwareRAG:
    def __init__(self, llm_api_key= None, embedder_model="all-MiniLM-L6-v2" ):
        self.embedder = Embedder(embedder_model)
        self.chatbot = Chatbot(api_key=llm_api_key)
        
        self.cv_text = None
        self.cv_vector = None
        self.chat_history = []
        
        #similarity threshold
        self.threshold = 0.05
        
        #Pdf ingestion
    def ingest_pdf(self, path: str):
        self.cv_text = load_pdf(path)
        self.cv_vector = self.embedder.embed([self.cv_text])[0]
        return "CV loaded successfully!"

        
    # history aware prompt rewriting
    def rewrite_history_aware_prompt(self, question: str):
        # format chat history
        history_text = ""
        for turn in self.chat_history:
            history_text += f"user: {turn['user']}\nAssistant: {turn['assistant']}\n"

        prompt = (
            "You are an intelligent assistant whose task is to rewrite the user's latest question "
            "into a standalone question. This standalone question should be fully self-contained, "
            "including any context necessary from the conversation so far.\n\n"
            "IMPORTANT:\n"
            "- If the user is asking about the conversation, previous questions, or wants a summary, "
            "rewrite it in a way that makes the request clear and answerable based on chat history. Otherwise just simply follow the previous instructions\n"
            "- Do not add or infer information; just make the question standalone.\n"
            "- Always rewrite the question in a clear, concise, and fully understandable way.\n\n"
            f"Chat History:\n{history_text}\n"
            f"Latest user question: {question}\n\n"
            "Standalone question:"
        )

        rewritten = self.chatbot.get_response(prompt)
        return rewritten.strip()

    
    #main answer function
    def answer(self, question: str):
        if len(self.chat_history) == 0:
            rewritten_question = question
        else:
            rewritten_question = self.rewrite_history_aware_prompt(question)
            
        #RAG logic
        #turn question into embedding
        question_vec = self.embedder.embed([rewritten_question])[0]
        
        #compute cosine similartiy with cv
        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


        sim_cv = cosine(question_vec, self.cv_vector)
        
        if sim_cv <self.threshold:
            answer = "i dont know."
            
            #save history
            self.chat_history.append({
                "user": question,
                "assistant": answer
            })
            
            # keep last 10 turns
            if len(self.chat_history) >10:
                self.chat_history.pop(0)
            
            return {
                "answer" :answer,
                "relevance": round(sim_cv, 3),
                "rewritten_question": rewritten_question,
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
            "6. Keep answers short and concise.Not more tha 3 lines,even that is when needed for great detail, the recuiter has to review hundred of CVs so keep the answers relevant and dont give out too much analysis.\n\n"
            f"CV Content:\n{self.cv_text}\n\n"
            f"Question: {rewritten_question}\n"
            "Answer as a recruiter:"
        )   
        
        response = self.chatbot.get_response(prompt)
        
        # Save full history
        self.chat_history.append({
            "user": question,
            "assistant": response
        })
        # keep last 10 turns
        if len(self.chat_history) >10:
            self.chat_history.pop(0)
        return {
            "answer": response,
            "rewritten_question": rewritten_question,
            "relevance": round(sim_cv, 3)
        }