Here’s the compiled version of the entire OmniRAG Assistant project, incorporating LangGraph-based agentic workflows, a modular monolithic architecture, hybrid retrieval, and self-hosted LLM integration.

📌 OmniRAG Assistant - Full System Design Document

A multi-modal, agentic knowledge companion that supports self-hosted LLMs, multi-source retrieval, and user-personalized AI interactions.

1️⃣ Functional Requirements

✅ User Authentication
✅ User Storage Bin (for links, files, personal context, etc.)
✅ Model Repository Interface (choose built-in models or connect API keys)
✅ Conversational AI (chat with models based on personal data)

2️⃣ Non-Functional Requirements

✅ Chat History & Contextual Memory
✅ Personalized AI Context
✅ Strict Data Isolation (per user)
✅ Modular Monolithic Architecture (with separate frontend & backend)

3️⃣ System Architecture

graph TD
    A[User Query] --> B{LangGraph Agent Router}
    B -->|Simple Query| C[Direct Retrieval (Qdrant + BM25)]
    B -->|Complex Query| D[Multi-Agent Workflow]
    
    D --> E[Text Agent (LangChain)]
    D --> F[Image Agent (CLIP)]
    D --> G[Audio Agent (Whisper)]
    D --> H[Web Agent (Google Search)]
    D --> I[Personal Context Agent]
    
    E --> J[Hybrid Search (Vector + Keyword)]
    F --> J
    G --> J
    H --> J
    I --> J

    J --> K[LangChain + LlamaIndex]
    K --> L[Final Response Generator]
    L --> M[User Output]

4️⃣ Tech Stack

Category	Technology
Frontend	Next.js (React)
Backend	FastAPI (Python)
Authentication	OAuth
Database	PostgreSQL
Vector Store	Qdrant
Text Processing	LangChain
LLMs	LLaMA3, Mistral, Falcon (via Ollama)
Image Processing	CLIP / BLIP
Audio Processing	Whisper
Web Scraping	Scrapy, Playwright
Caching	Redis
Workflow Orchestration	LangGraph
Deployment	Docker (Monolithic Stack)

5️⃣ API Design

🔑 Authentication

Endpoint	Method	Description
/auth/login	POST	User login via OAuth / JWT
/auth/register	POST	New user registration
/auth/logout	POST	Logout user

📂 Storage Bin

Endpoint	Method	Description
/storage/upload	POST	Upload files (PDF, DOCX, etc.)
/storage/list	GET	List user files
/storage/delete/{id}	DELETE	Remove a file

🧠 Model Selection

Endpoint	Method	Description
/models/list	GET	Get available models
/models/add	POST	Add API key (OpenAI, Claude, etc.)
/models/select	POST	Choose an inference model

💬 Chat System

Endpoint	Method	Description
/chat/start	POST	Start a new conversation
/chat/send_message	POST	Send a query to the assistant
/chat/history	GET	Retrieve chat history

6️⃣ Agent-Based Workflows (LangGraph)

🔹 Direct Retrieval Workflow

from langgraph.graph import StateGraph
from langchain.tools import Tool

def retrieve_data(query):
    return vector_search.retrieve(query)  # Qdrant/BM25 search

graph = StateGraph()
graph.add_node("retrieve", retrieve_data)
graph.set_entry_point("retrieve")

executor = graph.compile()
result = executor.invoke({"query": "What files do I have?"})
print(result)

🔹 Multi-Agent Processing Workflow

from langgraph.graph import StateGraph
from langchain.agents import initialize_agent, AgentType

def query_image_analysis(image_url):
    return clip_model.extract_text(image_url)

def query_text_analysis(text):
    return langchain_model.summarize(text)

graph = StateGraph()
graph.add_node("analyze_image", query_image_analysis)
graph.add_node("analyze_text", query_text_analysis)

graph.add_edge("analyze_image", "analyze_text")  # Chain tasks together
graph.set_entry_point("analyze_image")

executor = graph.compile()
result = executor.invoke({"image_url": "user_image.png"})
print(result)

🔹 Adaptive RAG Workflow

from langgraph.graph import StateGraph
from langchain.tools import Tool
from langchain.llms import Ollama

def decide_query_type(query):
    if "image" in query:
        return "image_processing"
    elif "summarize" in query:
        return "text_processing"
    return "direct_retrieval"

graph = StateGraph()
graph.add_node("decide", decide_query_type)
graph.add_node("text_processing", lambda q: llm.summarize(q))
graph.add_node("image_processing", lambda img: clip_model.extract_text(img))
graph.add_node("direct_retrieval", retrieve_data)

graph.add_edge("decide", "text_processing", condition=lambda x: x=="text_processing")
graph.add_edge("decide", "image_processing", condition=lambda x: x=="image_processing")
graph.add_edge("decide", "direct_retrieval", condition=lambda x: x=="direct_retrieval")

graph.set_entry_point("decide")
executor = graph.compile()

result = executor.invoke({"query": "Summarize my last document"})
print(result)

7️⃣ Deployment Strategy

Monolithic Deployment:
✅ Dockerized with all dependencies.
✅ Runs LLMs locally via Ollama.
✅ Modularized backend (FastAPI) for easy expansion.

Local & Cloud Deployment Options:
✅ Local: Runs self-hosted LLaMA models.
✅ Cloud: Uses PostgreSQL, Qdrant, Redis for scalability.

8️⃣ Security Considerations

✅ User Data Isolation – No cross-user contamination.
✅ Role-Based Access Control – API restrictions per user.
✅ Data Encryption – Securely stored files & chat logs.
✅ Limited API Exposure – Only user-controlled keys allowed.

9️⃣ Future Enhancements

🚀 Fine-Tuned Personal AI Assistants – Train personalized LLMs for each user.
🚀 Browser Extension – Save web pages directly into storage bins.
🚀 Mobile App Support – Native apps for iOS & Android.

🎯 Next Steps

✅ MVP Development Plan
	1.	Implement LangGraph-based agent workflows
	2.	Develop hybrid search (Qdrant + BM25)
	3.	Build API layer (FastAPI)
	4.	Integrate local Ollama models
	5.	Develop UI (React/Next.js)
	6.	Enable chat memory & context storage
	7.	Deploy as a single Docker service

🚀 OmniRAG Assistant is now fully designed and ready for development!

Let me know if you need refinements or if you’d like to start with code implementation. 🚀