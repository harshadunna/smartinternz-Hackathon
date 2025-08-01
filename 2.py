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

# --- Core RAG & Quiz Functions ---
def build_rag_pipeline(pdf_bytes):
    st.toast("Processing PDF...", icon="📄")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    text_chunks = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
    if not text_chunks:
        st.error("Could not extract substantial text chunks. The PDF might be image-based or have an unusual format.")
        return None

    st.toast("Generating embeddings...", icon="🧠")
    embeddings = EMBEDDING_MODEL.encode(text_chunks, convert_to_tensor=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.toast("PDF processed successfully!", icon="✅")
    return {"index": index, "chunks": text_chunks}

def get_rag_response(user_question, rag_pipeline):
    index = rag_pipeline['index']
    chunks = rag_pipeline['chunks']
    question_embedding = EMBEDDING_MODEL.encode([user_question])
    D, I = index.search(np.array(question_embedding), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

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
    chunks = rag_pipeline['chunks']
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
        json_str = re.search(r'```json\n(.*)\n```', response.text, re.DOTALL).group(1)
        quiz_questions = json.loads(json_str)
        return quiz_questions
    except Exception as e:
        st.error(f"Failed to generate or parse the quiz. Error: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="StudyMate", layout="wide")
st.markdown("""
    <style>
        .user-msg { background-color: #e0f7fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
        .bot-msg { background-color: #f3e5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
        .quiz-question { font-weight: bold; font-size: 1.1rem; }
        .stChatMessage { margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("📚 StudyMate - AI-Powered Academic Assistant")
st.write("Upload a PDF and interact with it using powerful AI.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Extracting knowledge..."):
            st.session_state.rag_pipeline = build_rag_pipeline(uploaded_file.getvalue())

    if st.session_state.rag_pipeline:
        st.success("✅ Document processed successfully!")
        st.markdown("---")
        if st.button("🧠 Generate Quiz"):
            with st.spinner("Creating quiz..."):
                quiz = generate_quiz(st.session_state.rag_pipeline)
                if quiz:
                    st.session_state.messages.append({"role": "assistant", "content": "Here's your quiz!", "quiz": quiz})
                else:
                    st.warning("Could not create quiz from this document.")

# --- Chat Interface ---
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"<div class='user-msg'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{message['content']}</div>", unsafe_allow_html=True)
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    st.info(source)
        if "quiz" in message:
            for q_idx, q in enumerate(message["quiz"]):
                unique_key = f"q_{i}_{q_idx}"
                st.markdown(f"<div class='quiz-question'>Q{q_idx+1}: {q['question']}</div>", unsafe_allow_html=True)
                chosen = st.radio("Options", q['options'], key=unique_key, label_visibility="collapsed")
                if st.button(f"Check Answer for Q{q_idx + 1}", key=f"check_{unique_key}"):
                    if chosen == q['answer']:
                        st.success(f"Correct! ✅ {q['answer']}")
                    else:
                        st.error(f"Incorrect. ❌ Correct Answer: {q['answer']}")
                st.markdown("---")

if prompt := st.chat_input("Ask a question from the PDF..."):
    if not st.session_state.rag_pipeline:
        st.warning("Please upload and process a document first.", icon="⚠️")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

        with st.spinner("Generating response..."):
            answer, sources = get_rag_response(prompt, st.session_state.rag_pipeline)
            st.markdown(f"<div class='bot-msg'>{answer}</div>", unsafe_allow_html=True)
            with st.expander("📚 View Sources"):
                for source in sources:
                    st.info(source)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})