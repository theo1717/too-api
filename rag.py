# rag.py
import os
import numpy as np
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# ---- CONFIG ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GROQ_API_KEY:
    raise Exception("Erro: GROQ_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

MODEL_EMBED = "llama-3.2-1b"  # modelo leve para embeddings
MODEL_LLM = "llama-3.2-1b"    # mesmo modelo para resposta


# -------- FUNÇÃO 1: gerar embedding via API --------
async def get_embedding(text: str):
    response = groq_client.embeddings.create(
        model=MODEL_EMBED,
        input=text
    )
    return response.data[0].embedding


# -------- FUNÇÃO 2: buscar top-k embeddings mais parecidos no Mongo --------
async def search_similar_docs(query_embedding, k=3):
    results = []

    async for doc in collection_embeddings.find():
        doc_emb = np.array(doc["embedding"])
        q_emb = np.array(query_embedding)

        # cosine similarity
        similarity = np.dot(q_emb, doc_emb) / (
            np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)
        )

        results.append((similarity, doc["text"]))

    # ordenar por similaridade
    results.sort(key=lambda x: x[0], reverse=True)

    top_texts = [r[1] for r in results[:k]]

    return "\n".join(top_texts)


# -------- FUNÇÃO 3: gerar resposta com contexto --------
async def rag_answer(query: str):

    # 1. Embedding da pergunta
    query_emb = await get_embedding(query)

    # 2. Busca no MongoDB
    context = await search_similar_docs(query_emb)

    # 3. Monta prompt final
    prompt = f"""
    Você é um assistente útil.

    CONTEXTO RELEVANTE:
    {context}

    PERGUNTA:
    {query}

    Responda de forma clara e objetiva.
    """

    # 4. Gera resposta
    response = groq_client.chat.completions.create(
        model=MODEL_LLM,
        messages=[
            {"role": "system", "content": "Você é um assistente útil."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response.choices[0].message["content"]
