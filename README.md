📘 NLP Buddy

NLP Buddy is an AI-powered assistant designed to help users interact with Natural Language Processing (NLP) content efficiently.
It combines Retrieval-Augmented Generation (RAG), OCR pipelines, and a chatbot with conversational memory to deliver context-aware, grounded, and persistent responses.

🚀 Features

🔎 Retrieval-Augmented Generation (RAG): Retrieves relevant chunks from NLP documents using FAISS as the vector store.

📄 PDF & OCR Support:

pdfplumber for extracting structured text from PDFs.

OCR pipeline to handle scanned/image-based documents.

🧠 Conversational Memory Buffer: Keeps track of past conversations for context-aware responses.

🤖 LLM-Powered Chatbot: Uses Gemini LLM for generating natural responses.

🔤 Embeddings from Hugging Face: Provides semantic search and document retrieval.

🛠️ Tech Stack

LLM: Google Gemini

Embeddings: Hugging Face models

Vector Database: FAISS

Document Processing: pdfplumber, pytesseract (OCR)

Frameworks: LangChain, Python

⚙️ Workflow

Document Ingestion: PDFs are processed using pdfplumber, OCR extracts text from scanned docs.

Embedding & Indexing: Hugging Face embeddings are stored in FAISS for similarity search.

User Query: The chatbot receives a query.

RAG Process: FAISS retrieves relevant document chunks.

Response Generation: The Gemini LLM generates an answer grounded in retrieved context.

Memory Handling: A memory buffer stores conversation history for continuity.
