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
# 1. Page Config + Global Style
# ===============================
st.set_page_config(
    page_title="📚 Ask My Documents",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Chat bubbles */
section[data-testid="stChatMessage"] {
    background-color: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 10px;
}

/* User bubble */
section[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    background-color: rgba(0, 123, 255, 0.18);
}

/* Assistant bubble */
section[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background-color: rgba(40, 167, 69, 0.18);
}

/* Chat input */
div[data-testid="stChatInput"] textarea {
    border-radius: 12px;
    padding: 12px;
}

/* Expander */
details {
    background-color: rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 2. Sidebar
# ===============================
with st.sidebar:
    st.title("📚 Ask My Docs")
    st.markdown("""
    Chat with your **private documents** using  
    **Groq + LangChain + FAISS**
    """)
    st.divider()
    st.markdown("**Supported formats**")
    st.markdown("- PDF\n- DOCX\n- TXT")
    st.divider()
    st.caption("⚡ LLaMA 3.1 via Groq")

# ===============================
# 3. Load Environment Variables
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# ===============================
# 4. Extract Text from Documents
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
            with open(file_path, "r", encoding="utf-8") as f:
                text_data += f.read()

        if text_data.strip():
            docs.append({"content": text_data, "source": file})

    return docs

# ===============================
# 5. Vectorstore
# ===============================
@st.cache_resource(show_spinner=False)
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts, metadatas = [], []
    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        texts.extend(chunks)
        metadatas.extend([{"source": doc["source"]}] * len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# ===============================
# 6. Conversational Chain
# ===============================
@st.cache_resource(show_spinner=False)
def build_chat_chain(_vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=GROQ_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# ===============================
# 7. Load & Index Documents
# ===============================
st.title("📄 Ask My Documents")
st.caption("Chat with PDFs, DOCX & TXT files")

if not os.path.exists("data") or not os.listdir("data"):
    st.warning("📂 Place your documents inside the `data/` folder")
    st.stop()

with st.spinner("📖 Indexing documents..."):
    documents = extract_text_with_sources("data")
    vectorstore = build_vectorstore(documents)
    qa_chain = build_chat_chain(vectorstore)

st.success(f"✅ Loaded {len(documents)} documents")

# ===============================
# 8. Chat Interface
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask something about your documents...")

# Display history
for user, bot in st.session_state.chat_history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(bot)

# Handle new query
if query:
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.spinner("🤖 Thinking..."):
        result = qa_chain({"question": query})
        answer = result["answer"]
        sources = [
            doc.metadata.get("source", "Unknown")
            for doc in result.get("source_documents", [])
        ]

    if not sources:
        answer = "❌ I couldn’t find that information in your documents."

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(answer)

        if sources:
            with st.expander("📄 Sources used"):
                for src in sorted(set(sources)):
                    st.markdown(f"- `{src}`")

    st.session_state.chat_history.append((query, answer))
