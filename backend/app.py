from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import query_relevant_chunks, generate_response
from rag_engine import ensure_collection_ready

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    print("Ensuring collection: ")
    ensure_collection_ready("/app/GGTexts")

    print("✅ Received question:", query.question)
    
    try:
        passages = query_relevant_chunks(query.question)
        print("✅ Retrieved passages:\n", passages)
        
        augmented_prompt = f"""
        The following are relevant passages:
        <retrieved-data>
        {passages}
        </retrieved-data>

        Here's the original user prompt, answer in the same language with help of the retrieved passages:
        <user-prompt>
        {query.question}
        </user-prompt>
        """
        print("📦 Sending prompt to Ollama...")

        answer = generate_response(augmented_prompt)
        print("✅ Got response from Ollama:", answer)

        return {"answer": answer}

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}
