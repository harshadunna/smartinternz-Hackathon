# studymate_app.py

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

# --- API and Model Configuration ---
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    GEMINI_MODEL = genai.GenerativeModel('gemini-1.5-flash-latest')
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.error(f"Error configuring Google API: {e}. Make sure your GOOGLE_API_KEY is set in a .env file.", icon="🚨")
    st.stop()


# --- Core RAG & Quiz Functions (Now Implemented) ---

def build_rag_pipeline(pdf_bytes):
    """Builds the RAG pipeline from PDF content."""
    st.toast("Processing PDF...", icon="📄")
    # 1. Extract text from PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    # 2. Split text into manageable chunks (by paragraphs)
    text_chunks = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
    if not text_chunks:
        st.error("Could not extract substantial text chunks. The PDF might be image-based or have an unusual format.")
        return None

    # 3. Generate embeddings
    st.toast("Generating embeddings...", icon="🧠")
    embeddings = EMBEDDING_MODEL.encode(text_chunks, convert_to_tensor=False)

    # 4. Create and populate FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.toast("PDF processed successfully!", icon="✅")
    return {"index": index, "chunks": text_chunks}


def get_rag_response(user_question, rag_pipeline):
    """Generates a response using the RAG pipeline with Gemini."""
    index = rag_pipeline['index']
    chunks = rag_pipeline['chunks']

    # 1. Embed the user's question
    question_embedding = EMBEDDING_MODEL.encode([user_question])

    # 2. Perform similarity search
    D, I = index.search(np.array(question_embedding), k=3)  # Retrieve top 3 chunks
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # 3. Construct prompt and call Gemini
    prompt = f"""
    You are an expert academic assistant. Answer the user's question based ONLY on the provided context.
    If the answer is not found in the context, state that clearly.

    CONTEXT:
    {context}

    QUESTION:
    {user_question}

    ANSWER:
    """
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text, retrieved_chunks
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Sorry, I couldn't generate a response.", []


def generate_quiz(rag_pipeline):
    """Generates a multiple-choice quiz using Gemini."""
    chunks = rag_pipeline['chunks']
    # Select a few random chunks to base the quiz on
    sample_chunks = np.random.choice(chunks, size=min(3, len(chunks)), replace=False)
    context = "\n\n".join(sample_chunks)

    prompt = f"""
    Based on the following text, generate exactly 2 multiple-choice questions.
    Format the output as a valid JSON list of objects. Each object must have these keys: "question", "options" (a list of 4 strings), and "answer" (the correct option string).

    TEXT:
    {context}

    JSON_OUTPUT:
    """
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        # Clean and parse the JSON response
        json_str = re.search(r'```json\n(.*)\n```', response.text, re.DOTALL).group(1)
        quiz_questions = json.loads(json_str)
        return quiz_questions
    except Exception as e:
        st.error(f"Failed to generate or parse the quiz. Error: {e}")
        return None


# --- Streamlit UI (Largely Unchanged) ---

st.set_page_config(page_title="StudyMate", layout="wide")
st.title("StudyMate: Your AI-Powered Academic Assistant 📚")
st.write("Upload your PDF and let Gemini help you study!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Building knowledge base..."):
                st.session_state.rag_pipeline = build_rag_pipeline(uploaded_file.getvalue())

    if st.session_state.rag_pipeline:
        st.success("Document ready!")
        st.markdown("---")
        if st.button("🧠 Quiz Me!"):
            with st.spinner("Generating your quiz..."):
                quiz = generate_quiz(st.session_state.rag_pipeline)
                if quiz:
                    st.session_state.messages.append({"role": "assistant", "content": "Here's your quiz!", "quiz": quiz})
                else:
                    st.warning("Could not generate a quiz from this document.")

# Main Chat Interface
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(source)
        if "quiz" in message:
            for q_idx, q in enumerate(message["quiz"]):
                unique_key = f"q_{i}_{q_idx}"
                st.subheader(f"Question {q_idx + 1}: {q['question']}")
                chosen_option = st.radio("Options", q['options'], key=unique_key, label_visibility="collapsed")
                if st.button(f"Check Answer for Q{q_idx + 1}", key=f"check_{unique_key}"):
                    if chosen_option == q['answer']:
                        st.success(f"Correct! The answer is {q['answer']}.")
                    else:
                        st.error(f"Incorrect. The correct answer is {q['answer']}.")
                st.markdown("---")


if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.rag_pipeline:
        st.warning("Please upload and process a document first.", icon="⚠️")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_rag_response(prompt, st.session_state.rag_pipeline)
                st.markdown(answer)
                with st.expander("View Sources"):
                    for source in sources:
                        st.info(source)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})