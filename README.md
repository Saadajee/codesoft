# RAG Chatbot API – CV Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot API that answers questions based on a candidate’s CV.(Can be modfied for any type of scenario and documentations as per requirements). Built with **FastAPI**, **Groq API** for embeddings and completions, and Docker for deployment.  


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


## Setup instructions

First create a virtual envirement using python or conda to operate the new libraries in, After wards use the following commands:
- To install the dependencies reuqired:
```
#for windows
python3.11 -m venv rag_env 
rag_env\Scripts\activate
#for conda
conda create -n rag_env python=3.11
conda activate rag_env
```

- To install the dependencies reuqired:
```
pip install -r requirements.txt
```

- Make sure to create a .env file and put in your GROQ_API_KEY to be accessed.

- To run the FastAPI backend:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- Access your open API port through [http://127.0.0.1:8000](http://127.0.0.1:8000)




