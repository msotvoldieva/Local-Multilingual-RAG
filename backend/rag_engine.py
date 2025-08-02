import os
import uuid
import requests
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import time
from qdrant_client.http.exceptions import UnexpectedResponse

# === CONFIG ===
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
COLLECTION_NAME = "Galileo"
QDRANT_URL = "http://qdrant:6333"  # updated to Docker service name
OLLAMA_URL = "http://ollama:11434"

# === Init Embedding Model and Qdrant ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = QdrantClient(url=QDRANT_URL)

def wait_for_qdrant(timeout=30):
    print("⏳ Waiting for Qdrant to be ready...")
    for _ in range(timeout):
        try:
            client.get_collections()
            print("✅ Qdrant is ready.")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("❌ Qdrant did not start in time.")


def ensure_collection_ready(folder_path="GGTexts", collection_name=COLLECTION_NAME):
    wait_for_qdrant()
    if client.collection_exists(collection_name=collection_name):
        print(f"✅ Collection '{collection_name}' already exists.")
        return
    
    print(f"📂 Collection '{collection_name}' not found. Creating and populating it...")

    # 1. Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print("✅ Created collection.")

    # 2. Load and split documents
    chunks = create_chunks(folder_path)
    print(f"✂️ Created {len(chunks)} chunks.")

    # 3. Store in Qdrant
    store_chunks(chunks)
    print(f"📦 Stored chunks into Qdrant collection '{collection_name}'.")


# === Document Loading + Chunking ===
def load_documents(folder_path: str):
    print("Loading .txt files...")
    documents = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            path = os.path.join(folder_path, file_name)
            loader = TextLoader(path, encoding='utf-8')
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["title"] = file_name.replace(".txt", "")
                doc.metadata["source"] = file_name
            documents.extend(loaded_docs)
    return documents

def create_chunks(folder: str):
    docs = load_documents(folder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# === Embedding and Qdrant Storage ===
def generate_embeddings(text: str) -> List[float]:
    return embedding_model.encode(
        text,
        convert_to_tensor=True,
        normalize_embeddings=True
    ).tolist()

def store_chunks(chunks):
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        payload = {
            "content": chunk.page_content,
            **chunk.metadata
        }
        embedding = generate_embeddings(chunk.page_content)
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[PointStruct(id=chunk_id, vector=embedding, payload=payload)]
        )

# === Querying & Generation ===
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def query_relevant_chunks(user_prompt: str) -> str:
    task = 'Given a user query, retrieve relevant passages that answer the query'
    querie = get_detailed_instruct(task, user_prompt)
    query_embedding = generate_embeddings(querie)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=7
    )

    return "\n".join([
        f"- From: {pt.payload.get('title', 'Unknown')} \n{pt.payload.get('content', '')}\n"
        for pt in results.points
    ])

def generate_response(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "wizardlm2:7b", "prompt": prompt, "stream": False}
        )
        print("📥 Ollama raw response:", response.text)

        data = response.json()
        if "response" not in data:
            return f"[⚠️ Ollama error] Unexpected format: {data}"
        return data['response']

    except Exception as e:
        print("❌ Failed to get response from Ollama:", e)
        return f"[❌ Ollama failure] {str(e)}"


# === Optional: For CLI local testing ===
def interactive_cli():
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        try:
            passages = query_relevant_chunks(query)
            augmented_prompt = f"""
            The following are relevant passages:
            <retrieved-data>
            {passages}
            </retrieved-data>

            Here's the original user prompt, answer in the same language with help of the retrieved passages:
            <user-prompt>
            {query}
            </user-prompt>
            """
            response = generate_response(augmented_prompt)
            print("📘 Answer:", response, "\n")
        except Exception as e:
            print("[✗] Failed to generate answer:", str(e))