# app.py
import streamlit as st
from langchain.vectorstores import FAISS
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from typing import Any, List, Optional
from pdf2image import convert_from_path
import google.generativeai as genai
from IPython.display import Markdown, display
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import os
from metric import compute_exact_match, compute_f1

load_dotenv()
api_key = os.getenv("API_KEY_2")
genai.configure(api_key=api_key)

llm_model = genai.GenerativeModel("gemini-1.5-flash-8b-latest")

DB_FAISS_PATH = "C:/Users/20128/Desktop/NTI_PROJECT/nlp_vectorstore"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load FAISS DB
vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert NLP assistant. Your primary goal is to give a detailed, well-structured answer to the user's question using the provided context.\n\n"
        "### Instructions:\n"
        "1. *Focus on the context first.* Extract every relevant piece of information from the context and clearly explain it.\n"
        "2. *Be thorough.* Your answer should be at least 3–5 sentences long (or longer if needed) and should not be too brief.\n"
        "3. *Organize the answer.* Use bullet points, short paragraphs, or step-by-step explanations when appropriate.\n"
        "4. *Add value.* After covering the context, provide additional clarifying examples or explanations from your own knowledge.\n"
        "5. *Be precise.* Do not contradict the context or invent false information. If the context does not contain the answer, say:\n"
        "   'I could not find this information in the provided documents.'\n\n"
        "### Context:\n{context}\n\n"
        "### Question:\n{question}\n\n"
        "### Detailed Answer:"

    )
)

intent_prompt = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a strict keyword detector.\n"
        "Respond with ONLY 'yes' if the user's question explicitly contains the word "
        "'summary', 'summarize', 'summaries', or any closely related form.\n"
        "If no such word appears, respond with 'no'.\n\n"
        "Question: {question}"
    )
)

summarization_prompt = PromptTemplate(
    input_variables=["context"],
   template=(
        "You are an expert summarizer. Your job is to create a well-organized summary of the following text.\n\n"
        "### Instructions:\n"
        "1. Read the entire text carefully.\n"
        "2. Extract all key points and main ideas.\n"
        "3. Present the summary in *clear, concise bullet points*.\n"
        "4. Use short, informative sentences.\n"
        "5. Maintain the original meaning and include all important details.\n\n"
        "### Text:\n{context}\n\n"
        "### Bullet-Point Summary:"
    )
)


class GeminiLLM(LLM):
    model: Any = None  # The GenerativeModel instance

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini-llm"

# Initialize custom LLM
llm = GeminiLLM(model=llm_model)


# ✅ Memory for conversation context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# ✅ Build Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": rag_prompt},
)

intent_chain = LLMChain(llm=llm, prompt=intent_prompt, output_key="intent")

# 📝 Summarization prompt

summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt, output_key="summary")

print("✅ Gemini-based Conversational RAG Chain is ready!")


def ask_question(query):
    # Step 1: Detect if this is a summarization request
    intent = intent_chain.run({"question": query}).strip().lower()

    # Step 2: Run normal QA chain
    result = qa_chain.invoke({"question": query})
    answer = result["answer"]
    sources = result["source_documents"]

    # Step 3: If summarization is requested → summarize retrieved docs
    summary = None
    if "yes" in intent:
        retrieved_text = "\n\n".join([doc.page_content for doc in sources])
        summary = summarization_chain.run({"context": retrieved_text}).strip()

    return answer,summary



# Example evaluation
# evaluation_data = [
  
#     {
#     "question": "What is LSTM?",
#     "ground_truth": "LSTM (Long Short-Term Memory) is a special type of Recurrent Neural Network (RNN) architecture designed to learn long-term dependencies in sequential data. Proposed by Hochreiter & Schmidhuber (1997). Unlike vanilla RNNs, it has an internal memory (cell state) that can carry information across many time steps. It uses gates (forget, input, output) to control the flow of information."
#     },

#     {
#         "question": "Advantages of LSTM?",
#         "ground_truth": """1. Solves the Vanishing Gradient Problem
#             •Can preserve information over long sequences.
#             •Unlike vanilla RNNs, LSTM doesn’t “forget” quickly.
#             2. Captures Long-Term Dependencies
#             •Keeps track of context across many time steps.
#             •Useful in tasks where earlier input influences much later output (e.g., machine translation).
#             3. Selective Memory with Gates
#             •Forget, input, and output gates act as filters.
#             •The network decides what to keep, update, or discard.
#             4. Better Accuracy in Sequence Tasks
#             •Proven to outperform vanilla RNNs in NLP, speech recognition, and time-series 
#             forecasting.
#             5. Flexible for Different Data Types
#             •Works with text, speech, video, stock prices, sensor data, etc.
#             6. Well-Researched & Widely Used
#             •Huge community support, tutorials, and pre-trained models available.""" 
#     },
#     {
#         "question": "What is GRU?",
#         "ground_truth": """•GRU is a gated architecture like LSTM.
#             •It controls information flow with fewer gates.
#             •Unlike LSTM, it has no separate cell state → only 
#             maintains a hidden state.
#             GRU was designed to:
#             • Be faster and simpler than LSTM
#             • Use fewer gates (only 2 instead of 3)
#             • Have fewer parameters, which helps it train quicker
#             GRU Has Two Gates:
#             ➢ Update Gate
#             ➢ Reset Gate"""
#     },
#     {
#                     "question": "When to choose GRU over LSTM ?",
#                     "ground_truth": """1. When speed matters
#             •GRUs are faster to train )fewer gates → fewer parameters).
#             •Better for real-time applications like chatbots or online 
#             recommendation.
#             2. When resources are limited
#             •GRUs use less memory & computation.
#             •Good for mobile/embedded devices or when using smaller GPUs.
#             3. When data is small or moderate
#             •GRUs have lower risk of overfitting due to simpler architecture.
#             •More suitable for tasks with limited training data.
#             4. When sequences are not very long
#             •For short/medium sequences, GRUs perform as well as LSTMs, but 
#             more efficiently.
#             5. When experimenting quickly
#             •Faster training allows rapid prototyping and trying different 
#             architectures"""
#     },

#    {
#                 "question": "How OCR Works?",
#                 "ground_truth": """• Image Acquisition
#             Input can be scanned documents, photos, or PDFs
#             • Preprocessing
#             Noise reduction, binarization, resizing, deskewing
#             Improves image quality for better recognition
#             • Text Detection (Segmentation)
#             Locates regions in the image that contain text
#             Segments lines, words, and individual characters
#             • Character Recognition
#             Uses pattern matching or ML models to identify characters
#             Converts visual symbols into digital characters (A-Z, 0-9, etc.)
#             • Postprocessing
#             Spell check, error correction, language modeling
#             Formats the recognized text into structured output
#             • Output Generation
#             Text is saved in formats like TXT, DOCX, or searchable PDFs"""
#     },
# ]

# em_scores, f1_scores = [], []

# for sample in evaluation_data:
#     answer, _ = ask_question(sample["question"])
#     em = compute_exact_match(answer, sample["ground_truth"])
#     f1 = compute_f1(answer, sample["ground_truth"])
#     em_scores.append(em)
#     f1_scores.append(f1)

# print(f"Exact Match Accuracy: {sum(em_scores)/len(em_scores):.2f}")
# print(f"Average F1 Score: {sum(f1_scores)/len(f1_scores):.2f}") 