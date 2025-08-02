# 🧠 Local RAG Interface (Multilingual, Text-Only)

This is a **local Retrieval-Augmented Generation (RAG)** interface that lets users **upload `.txt` files** and ask questions in **multiple languages**. It combines powerful open-source tools like **Streamlit**, **FastAPI**, **Qdrant**, and **Ollama** to run a full RAG pipeline locally via Docker.

<p align="center">
  <img src="demo_screenshot.png" alt="Demo" width="500"/>
</p>

---

## 🚀 Features

- 📄 Upload `.txt` documents to the `GGTexts/` folder
- 💬 Ask questions in multiple languages
- ⚡ Fast, local, and private — runs entirely on your machine
- 🧠 Uses LangChain and SentenceTransformers for embedding + RAG pipeline
- 🐳 One-command startup with Docker Compose

---

## 🛠️ Tech Stack

| Layer        | Tool/Library       |
|--------------|--------------------|
| Frontend     | Streamlit          |
| API Server   | FastAPI            |
| Embeddings   | SentenceTransformers |
| LLM          | Ollama             |
| Vector Store | Qdrant             |
| Orchestration| LangChain          |
| Containerization | Docker Compose |

---

## 🧩 Project Structure

RAG/

├── backend/ # FastAPI + LangChain logic

├── frontend/ # Streamlit interface

├── GGTexts/ # Uploaded text resources

├── qdrant_storage/ # Qdrant vector DB storage (local volume)

├── docker-compose.yml # Multi-service setup

├── requirements.txt # Python dependencies

└── .gitignore # Clean repo config


---

## 💻 How to Run It Locally (Docker)

### 🔧 Prerequisites

- Docker + Docker Compose installed
- Ollama installed with a supported model downloaded (intfloat/multilingual-e5-large-instruct)
  ```bash
  ollama serve

``` bash
git clone https://github.com/msotvoldieva/Local-Multilingual-RAG.git

cd Local-Multilingual-RAG

docker-compose up --build
```
Open http://localhost:8501 in your browser for the Streamlit UI

**📂 Uploading Documents**

Place .txt files into the GGTexts/ folder.

Files are loaded and chunked when the app starts. 

It might take longer on the first run, when the collection is being created (the model used is multilingual and takes a while). 

You can change the model if needed. Subsequent runs on the same collection will start instantly.

**✨ Example Use Cases**

Upload multilingual articles or notes and query them

Personal knowledge base with local language support

Use offline for secure document Q&A

**🤝 Contributing**

Contributions, ideas, and issues are welcome!

1. Fork this repo
2. Create a new branch: git checkout -b feature/your-feature
3. Commit your changes: git commit -m "add your message"
4. Push to the branch: git push origin feature/your-feature
5. Open a Pull Request

🙌 Acknowledgements
LangChain
Qdrant
Ollama
Streamlit
Sentence Transformers
