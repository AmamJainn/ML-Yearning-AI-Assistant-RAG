"""
================================================================================
 ML Yearning AI Assistant — Premium Streamlit UI
--------------------------------------------------------------------------------
 A Retrieval-Augmented Generation (RAG) chat application built on top of the
 existing project logic (create_database.py + main.py):

   - PDF   : machine-learning-yearning.pdf
   - Split : RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
   - Embed : sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
   - Store : Chroma (persist_directory="chroma_db")
   - Retrieve : MMR search (k=4, fetch_k=10, lambda_mult=0.5)
   - LLM   : ChatMistralAI (mistral-small-2506)
   - Prompt: strict "answer only from context" system prompt

 All original RAG logic from main.py is preserved untouched — this file only
 wraps it in a polished, professional Streamlit interface.

 Developed by: Amam Jain
 LinkedIn: https://www.linkedin.com/in/amam-jain-216082386
================================================================================
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="ML Yearning AI Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

CHROMA_DIR = "chroma_db"
PDF_NAME = "machine-learning-yearning.pdf"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral-small-2506"

# ------------------------------------------------------------------------------
# Custom CSS — premium SaaS look & feel
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1150px;
        }

        /* ---------- Hero Section ---------- */
        .hero {
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #C026D3 100%);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 20px 45px rgba(124, 58, 237, 0.25);
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute;
            top: -60px;
            right: -60px;
            width: 220px;
            height: 220px;
            background: rgba(255,255,255,0.08);
            border-radius: 50%;
        }
        .hero-title {
            font-family: 'Poppins', sans-serif;
            font-size: 2.6rem;
            font-weight: 800;
            color: white;
            margin-bottom: 0.4rem;
            letter-spacing: -0.5px;
        }
        .hero-subtitle {
            font-size: 1.08rem;
            color: rgba(255,255,255,0.92);
            max-width: 680px;
            line-height: 1.6;
            margin-bottom: 1.4rem;
        }
        .hero-badges span {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            color: white;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.4rem;
            backdrop-filter: blur(6px);
        }

        /* ---------- Generic Cards ---------- */
        .card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            border: 1px solid #EEF0F6;
            box-shadow: 0 6px 18px rgba(17, 12, 46, 0.04);
            transition: all 0.25s ease;
            height: 100%;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 30px rgba(79, 70, 229, 0.12);
            border-color: #E0D9FF;
        }
        .card h4 {
            margin-top: 0;
            margin-bottom: 0.4rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: #1F2133;
        }
        .card p {
            color: #5B5F73;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 0;
        }

        /* ---------- Section headings ---------- */
        .section-title {
            font-family: 'Poppins', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #1F2133;
            margin: 1.8rem 0 0.9rem 0;
        }
        .section-sub {
            color: #7A7E92;
            font-size: 0.95rem;
            margin-top: -0.6rem;
            margin-bottom: 1.2rem;
        }

        /* ---------- Answer card ---------- */
        .answer-card {
            background: linear-gradient(180deg, #FAFAFF 0%, #FFFFFF 100%);
            border: 1px solid #E7E4FB;
            border-left: 5px solid #7C3AED;
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-top: 0.8rem;
            box-shadow: 0 8px 22px rgba(124, 58, 237, 0.07);
        }
        .source-chip {
            display: inline-block;
            background: #F3F0FF;
            color: #6D28D9;
            border: 1px solid #E0D3FF;
            padding: 0.15rem 0.6rem;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            box-shadow: 0 6px 16px rgba(124, 58, 237, 0.25);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 22px rgba(124, 58, 237, 0.35);
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #14132A;
        }
        section[data-testid="stSidebar"] * {
            color: #E5E3F5 !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            font-weight: 500;
        }

        /* ---------- Chat bubbles ---------- */
        .user-bubble {
            background: #4F46E5;
            color: white;
            padding: 0.75rem 1.1rem;
            border-radius: 16px 16px 4px 16px;
            max-width: 80%;
            margin-left: auto;
            margin-bottom: 0.6rem;
            font-size: 0.95rem;
        }

        /* ---------- Footer ---------- */
        .footer {
            text-align: center;
            padding: 1.6rem 0 0.6rem 0;
            color: #9497AA;
            font-size: 0.85rem;
            border-top: 1px solid #EEF0F6;
            margin-top: 2rem;
        }
        .footer a {
            color: #7C3AED;
            font-weight: 600;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }

        hr {
            border-color: #EEF0F6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Cached resource loaders — reuse existing RAG logic from main.py
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False)
def load_vectorstore(_embedding_model):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_embedding_model,
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatMistralAI(model=LLM_MODEL)


def build_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )


def get_retriever(vectorstore, k, fetch_k, lambda_mult):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def answer_question(question, retriever, llm, prompt):
    """Runs the exact same retrieval + generation flow as main.py."""
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": question})
    response = llm.invoke(final_prompt)

    return response.content, docs


# ------------------------------------------------------------------------------
# Resource / state initialization
# ------------------------------------------------------------------------------
chroma_ready = os.path.isdir(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0
mistral_key_present = bool(os.getenv("MISTRAL_API_KEY"))

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {question, answer, sources}

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ------------------------------------------------------------------------------
# Sidebar — navigation & settings
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📘 ML Yearning AI")
    st.markdown("Retrieval-Augmented Q&A over Andrew Ng's *Machine Learning Yearning*.")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "💬 Ask the Assistant",
            "⚙️ How It Works",
            "🧠 Model Information",
            "🛠️ Tech Stack",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Retrieval Settings")
    k_value = st.slider("Chunks to retrieve (k)", 2, 8, 4)
    fetch_k_value = st.slider("Candidate pool (fetch_k)", 4, 20, 10)
    lambda_mult = st.slider("Diversity (λ, MMR)", 0.0, 1.0, 0.5, 0.1)

    st.markdown("---")
    st.markdown("### System Status")
    st.write("📦 Vector DB:", "✅ Ready" if chroma_ready else "❌ Not found")
    st.write("🔑 Mistral Key:", "✅ Loaded" if mistral_key_present else "❌ Missing")
    st.write("💬 Queries this session:", st.session_state.total_queries)

    st.markdown("---")
    st.caption("Developed by **Amam Jain**")
    st.markdown(
        "[🔗 Connect on LinkedIn](https://www.linkedin.com/in/amam-jain-216082386)"
    )

# ------------------------------------------------------------------------------
# Reusable UI components
# ------------------------------------------------------------------------------
def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📘 ML Yearning AI Assistant</div>
            <div class="hero-subtitle">
                An intelligent Retrieval-Augmented Generation assistant trained on
                Andrew Ng's <em>Machine Learning Yearning</em>. Ask any question about
                the book and get precise, context-grounded answers — instantly.
            </div>
            <div class="hero-badges">
                <span>🔍 Semantic Search</span>
                <span>🧠 Mistral LLM</span>
                <span>📚 Chroma Vector DB</span>
                <span>⚡ Real-time Answers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using Streamlit, LangChain, Chroma &amp; Mistral AI<br/>
            Developed by <b>Amam Jain</b> &nbsp;|&nbsp;
            <a href="https://www.linkedin.com/in/amam-jain-216082386" target="_blank">
                LinkedIn Profile ↗
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_card(icon, title, desc):
    st.markdown(
        f"""
        <div class="card">
            <h4>{icon} {title}</h4>
            <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# PAGE: Home
# ------------------------------------------------------------------------------
if page == "🏠 Home":
    render_hero()

    st.markdown('<div class="section-title">✨ Key Features</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card(
            "🔍",
            "Semantic Retrieval",
            "Finds the most relevant passages from the book using MMR-based "
            "vector similarity search over Chroma.",
        )
    with c2:
        feature_card(
            "🧠",
            "Grounded Answers",
            "Answers are generated strictly from retrieved context — "
            "no hallucinated content, no guessing.",
        )
    with c3:
        feature_card(
            "⚡",
            "Fast &amp; Interactive",
            "Get instant, conversational answers with a clean chat-style "
            "interface built for real product use.",
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        feature_card(
            "📚",
            "Source Transparency",
            "Every answer can be inspected alongside the exact document "
            "chunks used to generate it.",
        )
    with c5:
        feature_card(
            "🎛️",
            "Tunable Retrieval",
            "Adjust k, fetch_k, and MMR diversity (λ) live from the sidebar "
            "to control answer precision.",
        )
    with c6:
        feature_card(
            "🛡️",
            "Reliable by Design",
            "The assistant explicitly says when an answer isn't present "
            "in the document, instead of making one up.",
        )

    st.markdown('<div class="section-title">📊 Session Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Vector DB Status", "Ready" if chroma_ready else "Missing")
    m2.metric("Embedding Model", "MiniLM-L6-v2")
    m3.metric("LLM Model", LLM_MODEL)
    m4.metric("Queries This Session", st.session_state.total_queries)

    st.info("👉 Head to **💬 Ask the Assistant** in the sidebar to start chatting with the document.")
    render_footer()

# ------------------------------------------------------------------------------
# PAGE: Ask the Assistant
# ------------------------------------------------------------------------------
elif page == "💬 Ask the Assistant":
    st.markdown('<div class="section-title">💬 Ask the Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Ask any question about <i>Machine Learning Yearning</i> '
        "and receive an answer grounded strictly in the book's content.</div>",
        unsafe_allow_html=True,
    )

    if not chroma_ready:
        st.error(
            f"⚠️ Vector database not found at `{CHROMA_DIR}/`. "
            "Please run `create_database.py` first to build the knowledge base."
        )
    elif not mistral_key_present:
        st.warning(
            "⚠️ `MISTRAL_API_KEY` not found in your environment. "
            "Add it to a `.env` file before asking questions."
        )

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_area(
            "Your question",
            placeholder="e.g. What is the difference between the dev set and the test set?",
            height=90,
        )
        col_a, col_b = st.columns([1, 5])
        with col_a:
            submitted = st.form_submit_button("🚀 Ask")
        with col_b:
            st.caption("Answers are generated only from the retrieved book content.")

    if submitted:
        if not query or not query.strip():
            st.warning("Please enter a question before submitting.")
        elif not chroma_ready:
            st.error("Cannot answer — vector database is missing. Build it first with `create_database.py`.")
        else:
            try:
                with st.spinner("🔎 Retrieving relevant passages and generating an answer..."):
                    embedding_model = load_embedding_model()
                    vectorstore = load_vectorstore(embedding_model)
                    llm = load_llm()
                    prompt = build_prompt()
                    retriever = get_retriever(vectorstore, k_value, fetch_k_value, lambda_mult)

                    start = time.time()
                    answer, sources = answer_question(query, retriever, llm, prompt)
                    elapsed = time.time() - start

                st.session_state.chat_history.append(
                    {"question": query, "answer": answer, "sources": sources, "time": elapsed}
                )
                st.session_state.total_queries += 1
                st.success(f"✅ Answer generated in {elapsed:.2f}s")
            except Exception as e:
                st.error(f"❌ Something went wrong while generating the answer: {e}")

    # Display latest answer prominently
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.markdown(f"**🙋 Question:** {latest['question']}")
        st.markdown(f"**🤖 Answer:** {latest['answer']}")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander(f"📄 View {len(latest['sources'])} retrieved source chunk(s)"):
            for i, doc in enumerate(latest["sources"], start=1):
                page_num = doc.metadata.get("page", "N/A")
                st.markdown(
                    f'<span class="source-chip">Chunk {i}</span> '
                    f'<span class="source-chip">Page {page_num}</span>',
                    unsafe_allow_html=True,
                )
                st.write(doc.page_content)
                st.markdown("---")

        m1, m2 = st.columns(2)
        m1.metric("Response Time", f"{latest['time']:.2f}s")
        m2.metric("Chunks Used", len(latest["sources"]))

    # Conversation history
    if len(st.session_state.chat_history) > 1:
        st.markdown('<div class="section-title">🕘 Conversation History</div>', unsafe_allow_html=True)
        for turn in reversed(st.session_state.chat_history[:-1]):
            with st.expander(f"🙋 {turn['question']}"):
                st.write(turn["answer"])

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Conversation History"):
            st.session_state.chat_history = []
            st.session_state.total_queries = 0
            st.rerun()

    render_footer()

# ------------------------------------------------------------------------------
# PAGE: How It Works
# ------------------------------------------------------------------------------
elif page == "⚙️ How It Works":
    st.markdown('<div class="section-title">⚙️ How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A Retrieval-Augmented Generation (RAG) pipeline in five steps.</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("1️⃣", "Document Ingestion", "The PDF (`machine-learning-yearning.pdf`) is loaded using PyPDFLoader and parsed into raw page-level documents."),
        ("2️⃣", "Chunking", "Text is split into overlapping chunks (size=1000, overlap=200) using RecursiveCharacterTextSplitter to preserve context across boundaries."),
        ("3️⃣", "Embedding", "Each chunk is converted into a dense vector using the `sentence-transformers/all-MiniLM-L6-v2` HuggingFace embedding model."),
        ("4️⃣", "Vector Storage", "Embeddings are persisted in a local Chroma vector database (`chroma_db/`) for fast similarity search."),
        ("5️⃣", "Retrieval + Generation", "At query time, MMR search retrieves the most relevant, diverse chunks, which are passed as context to the Mistral LLM to generate a grounded answer."),
    ]

    for icon, title, desc in steps:
        st.markdown(
            f"""
            <div class="card" style="margin-bottom:1rem;">
                <h4>{icon} {title}</h4>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_footer()

# ------------------------------------------------------------------------------
# PAGE: Model Information
# ------------------------------------------------------------------------------
elif page == "🧠 Model Information":
    st.markdown('<div class="section-title">🧠 Model Information</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔡 Embedding Model")
        st.write("**Name:** sentence-transformers/all-MiniLM-L6-v2")
        st.write("**Provider:** HuggingFace")
        st.write("**Purpose:** Converts text chunks into dense vector embeddings for semantic similarity search.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🤖 Language Model")
        st.write(f"**Name:** {LLM_MODEL}")
        st.write("**Provider:** Mistral AI")
        st.write("**Purpose:** Generates the final natural-language answer strictly from retrieved context.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔎 Retrieval Configuration</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Search Type", "MMR")
    m2.metric("k (results)", k_value)
    m3.metric("fetch_k (pool)", fetch_k_value)
    st.metric("Diversity (λ)", lambda_mult)

    with st.expander("ℹ️ What is MMR search?"):
        st.write(
            "Maximal Marginal Relevance (MMR) balances **relevance** and **diversity** "
            "when selecting chunks — reducing redundant results while keeping the "
            "most relevant passages, controlled by the λ (lambda) parameter."
        )

    render_footer()

# ------------------------------------------------------------------------------
# PAGE: Tech Stack
# ------------------------------------------------------------------------------
elif page == "🛠️ Tech Stack":
    st.markdown('<div class="section-title">🛠️ Technology Stack</div>', unsafe_allow_html=True)

    stack = [
        ("🐍", "Python", "Core programming language for the entire pipeline."),
        ("🦜", "LangChain", "Framework orchestrating document loading, splitting, retrieval, and prompting."),
        ("📄", "PyPDFLoader", "Extracts text content from the source PDF document."),
        ("✂️", "RecursiveCharacterTextSplitter", "Splits documents into overlapping, context-preserving chunks."),
        ("🔡", "HuggingFace Embeddings", "Generates dense vector representations (all-MiniLM-L6-v2)."),
        ("🗄️", "Chroma", "Local vector database used to store and query embeddings."),
        ("🤖", "Mistral AI (ChatMistralAI)", "Large language model used for grounded answer generation."),
        ("🎈", "Streamlit", "Powers this interactive, deployment-ready web interface."),
    ]

    cols = st.columns(2)
    for i, (icon, name, desc) in enumerate(stack):
        with cols[i % 2]:
            feature_card(icon, name, desc)

    render_footer()

# ------------------------------------------------------------------------------
# PAGE: About
# ------------------------------------------------------------------------------
elif page == "ℹ️ About":
    st.markdown('<div class="section-title">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <p>
            <b>ML Yearning AI Assistant</b> is a Retrieval-Augmented Generation (RAG)
            application that transforms Andrew Ng's <i>Machine Learning Yearning</i> into
            an interactive, queryable knowledge base. Instead of manually searching the
            book, users can simply ask natural-language questions and receive precise,
            context-grounded answers backed directly by the source text.
            </p>
            <p>
            The system combines document chunking, semantic embeddings, vector search,
            and large language model generation into a single, reliable pipeline —
            with a strict guardrail against hallucinated answers: if the information
            isn't in the book, the assistant says so.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">👤 About the Developer</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
            <h4>Amam Jain</h4>
            <p>
            Developer of the ML Yearning AI Assistant.<br/><br/>
            🔗 <a href="https://www.linkedin.com/in/amam-jain-216082386" target="_blank">
            www.linkedin.com/in/amam-jain-216082386</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_footer()
