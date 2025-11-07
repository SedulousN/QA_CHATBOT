import streamlit as st
import os
from PyPDF2 import PdfReader
from docx import Document
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ===============================
# 1. Load Environment Variables
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file. Please add it and restart.")
    st.stop()

# ===============================
# 2. Extract Text from Documents
# ===============================
def extract_text_with_sources(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        text_data = ""

        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text_data += page.extract_text() or ""
        elif file.endswith(".docx"):
            doc = Document(file_path)
            for para in doc.paragraphs:
                text_data += para.text + "\n"
        elif file.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text_data += f.read()

        if text_data.strip():
            docs.append({"content": text_data, "source": file})
    return docs

# ===============================
# 3. Build Vectorstore with Metadata
# ===============================
@st.cache_resource(show_spinner=False)
def build_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts, metadatas = [], []

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"]}] * len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

# ===============================
# 4. Build Conversational QA Chain
# ===============================
@st.cache_resource(show_spinner=False)
def build_chat_chain(_vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=GROQ_API_KEY)
    
    # Memory to store conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # 👈 REQUIRED for LangChain v0.2+
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        output_key="answer",  # 👈 Match with memory's output_key
    )
    return chain


# ===============================
# 5. Streamlit UI
# ===============================
st.set_page_config(page_title="📚 Ask My Documents", layout="centered")
st.title("📚 Ask My Documents Chatbot")
st.caption("Chat with your PDFs, DOCX, and TXT files — powered by Groq + LangChain")

if not os.path.exists("data") or len(os.listdir("data")) == 0:
    st.warning("Please place your documents inside the `data/` folder and reload.")
    st.stop()

# Build once
with st.spinner("📖 Reading and indexing your documents..."):
    documents = extract_text_with_sources("data")
    vectorstore = build_vectorstore(documents)
    qa_chain = build_chat_chain(vectorstore)
st.success("✅ Documents processed successfully!")

# ===============================
# 6. Chat Interface
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("### 💬 Chat")
query = st.chat_input("Ask something about your documents...")

# Display chat history
for user, bot in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)

# Handle user query
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("🤖 Thinking (Groq)..."):
        result = qa_chain({"question": query})
        answer = result["answer"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]

    # Handle out-of-context queries
    if not sources:
        answer = "Sorry, I couldn’t find anything about that in your documents."

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                st.markdown(", ".join(set(sources)))

    # Save chat
    st.session_state.chat_history.append((query, answer))
