# rag.py
import os
import numpy as np
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import logging

load_dotenv()

# ---- CONFIG ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GROQ_API_KEY:
    raise Exception("Erro: GROQ_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

# ---- CLIENTE GROQ ----
try:
    http_client = httpx.Client()  # configure proxies aqui se precisar
    groq_client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
except Exception as e:
    logging.error(f"Falha ao inicializar Groq: {e}")
    groq_client = None  # fallback para evitar crash

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

MODEL_EMBED = "llama-3.2-1b"
MODEL_LLM = "llama-3.2-1b"

# -------- FUNÇÃO 1: gerar embedding via API --------
async def get_embedding(text: str):
    if not groq_client:
        logging.warning("Groq client não disponível, retornando embedding vazio")
        return np.zeros(768)  # fallback: vetor nulo

    try:
        response = groq_client.embeddings.create(
            model=MODEL_EMBED,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(768)  # fallback

# -------- FUNÇÃO 2: buscar top-k embeddings mais parecidos no Mongo --------
async def search_similar_docs(query_embedding, k=3):
    results = []

    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc["embedding"])
            q_emb = np.array(query_embedding)
            similarity = np.dot(q_emb, doc_emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(doc_emb) + 1e-10
            )
            results.append((similarity, doc["text"]))

        results.sort(key=lambda x: x[0], reverse=True)
        top_texts = [r[1] for r in results[:k]]
        return "\n".join(top_texts)
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# -------- FUNÇÃO 3: gerar resposta com contexto --------
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    prompt = f"""
Você é um assistente útil.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma clara e objetiva.
"""

    if not groq_client:
        logging.warning("Groq client não disponível, retornando mensagem fallback")
        return "Desculpe, não consigo gerar resposta no momento."

    try:
        response = groq_client.chat.completions.create(
            model=MODEL_LLM,
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message["content"]
    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
