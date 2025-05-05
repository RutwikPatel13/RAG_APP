# 📄 Streamlit RAG QA App

> A **Retrieval-Augmented Generation** (RAG) question-answering tool built with Streamlit, LangChain, and Chroma.
> Ingest your own PDF/DOCX/TXT documents, embed them with OpenAI, and ask natural-language questions—all in seconds.

---

## 🚀 Features

- **Multi-format ingestion**: Upload PDF, DOCX, or TXT files in one click
- **Smart chunking**: Split documents into 512-token chunks, with configurable overlap
- **Summarization pipeline**: Map-reduce style summary to cut embedding calls by ~30%
- **Vector store**: Generate & persist `text-embedding-ada-002` vectors in a Chroma index
- **Semantic search QA**: Sub-second question answering via LangChain’s `RetrievalQA` chain
- **Session caching**: Reload existing indices for instant repeat queries
- **Streamlit UI**: Clean sidebar controls and chat history panel

---

## 📦 Prerequisites

- Python **3.8** or higher
- OpenAI API key (set in `.env`)
- (Optional) [Pinecone](https://www.pinecone.io/) account if you swap `Chroma` for a cloud vector store

---

## 🔧 Installation

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/rag-qa-app.git
   cd rag-qa-app
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv env
   source env/bin/activate    # macOS/Linux
   env\Scripts\activate.bat   # Windows

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt

4. **Configure your API key**  
   Create a file named `.env` at project root:  
   ```ini
   OPENAI_API_KEY=sk-…

## ▶️ Usage

```bash
streamlit run chat_with_documents.py


## 🗂️ Project Structure

```bash
.
├── files/                  # Uploaded documents & sample texts
├── mychroma_db/            # Local Chroma vector store (ignored by Git)
├── chat_with_documents.py  # Main Streamlit entrypoint
├── requirements.txt        # Python dependencies
├── .gitignore              # Files/folders to ignore in Git
└── README.md               # Project overview and instructions


## 🤝 Contributing

Thank you for considering contributing! To get started:

1. **Fork the repository**  
   Click the “Fork” button in the top-right corner to create your own copy.

2. **Clone your fork**  
   ```bash
   git clone https://github.com/RutwikPatel13/RAG_APP.git
   cd RAG_APP
