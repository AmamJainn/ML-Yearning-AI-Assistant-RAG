# 📘 ML Yearning AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot built on top of Andrew Ng's **Machine Learning Yearning**. Ask any question about the book and get precise, context-grounded answers — powered by semantic search and a large language model.

🔗 **Live Demo:** [ml-yearning-ai-assistant-rag-xtwuayjygxfnbo3vngvjkd.streamlit.app](https://ml-yearning-ai-assistant-rag-xtwuayjygxfnbo3vngvjkd.streamlit.app/)

---

## ✨ Features

- 🔍 **Semantic Retrieval** — finds the most relevant passages from the book using MMR-based vector similarity search
- 🧠 **Grounded Answers** — answers are generated strictly from retrieved context, no hallucinations
- ⚡ **Fast & Interactive** — clean, premium Streamlit chat interface
- 📚 **Source Transparency** — inspect the exact document chunks used to generate each answer
- 🎛️ **Tunable Retrieval** — adjust `k`, `fetch_k`, and MMR diversity (λ) live from the sidebar

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Document Loading | `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | `Chroma` |
| LLM | `ChatMistralAI` (mistral-small-2506) |
| Orchestration | `LangChain` |
| UI | `Streamlit` |

---

## ⚙️ How It Works

1. **Document Ingestion** — `machine-learning-yearning.pdf` is loaded and parsed into page-level documents.
2. **Chunking** — text is split into overlapping chunks (`chunk_size=1000`, `chunk_overlap=200`) to preserve context.
3. **Embedding** — each chunk is converted into a dense vector using the MiniLM embedding model.
4. **Vector Storage** — embeddings are persisted in a local Chroma vector database (`chroma_db/`).
5. **Retrieval + Generation** — at query time, MMR search retrieves the most relevant, diverse chunks, which are passed as context to the Mistral LLM to generate a grounded answer.

---

## 📂 Project Structure

```
├── app.py                        # Streamlit UI (main entry point)
├── create_database.py            # Builds the Chroma vector database from the PDF
├── main.py                       # CLI version of the RAG pipeline
├── machine-learning-yearning.pdf # Source document
├── chroma_db/                    # Persisted vector store
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables (not committed)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your-mistral-api-key
```

### 4. Build the vector database (first-time only)

```bash
python create_database.py
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the project to GitHub — **exclude** `.venv/` and `.env` (see `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the main file as `app.py`.
4. Add your API key under **App Settings → Secrets**:
   ```toml
   MISTRAL_API_KEY = "your-mistral-api-key"
   ```
5. Deploy 🚀

---

## 👤 About the Developer

**Amam Jain**

🔗 [LinkedIn — amam-jain-216082386](https://www.linkedin.com/in/amam-jain-216082386)

---

## 📄 License

This project is for educational purposes. The source PDF (*Machine Learning Yearning* by Andrew Ng) is used under its original terms and is freely distributed by the author for personal/educational use.
