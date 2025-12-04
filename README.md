# RAG Chatbot API – CV Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot API that answers questions based on a candidate’s CV.  
Built with **FastAPI**, **Groq API** for embeddings and completions, and Docker for deployment.  


## Features

- Ask questions about a candidate’s CV.
- Context-aware answers using **last 10 Q&A** memory.
- Simple API endpoints for integration.
- Fully containerized with **Docker**.


## Libraries involved

- **Backend:** Python, FastAPI, Uvicorn
- **Embeddings & LLM:** Groq API (`groq` Python SDK)
- **PDF Parsing:** PyPDF2
- **Environment:** Docker, python-dotenv
- **Other:** Numpy, Requests



