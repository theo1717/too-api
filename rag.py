# rag_debug.py
import os
import numpy as np
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import logging

logging.basicConfig(level=logging.INFO)

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
    http_client = httpx.Client()
    groq_client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
except Exception as e:
    logging.error(f"Falha ao inicializar Groq: {e}")
    groq_client = None

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

MODEL_EMBED = "text-embedding-3-small"       # embeddings
MODEL_LLM = "llama-3.1-8b-instant"      # LLM

# -------- FUNÇÃO 1: gerar embedding via API --------
async def get_embedding(text: str, fallback_size=1024):
    if not groq_client:
        logging.warning("Groq client não disponível, retornando embedding vazio")
        return np.zeros(fallback_size)

    try:
        response = groq_client.embeddings.create(model=MODEL_EMBED, input=text)
        embedding = np.array(response.data[0].embedding)
        logging.info(f"Embedding gerado, tamanho: {embedding.shape}")
        return embedding
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return np.zeros(fallback_size)

# -------- FUNÇÃO 2: buscar top-k embeddings mais parecidos no Mongo --------
async def search_similar_docs(query_embedding, k=3):
    results = []
    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc["embedding"])
            if doc_emb.shape != query_embedding.shape:
                logging.warning(f"Tamanho do embedding do doc ({doc_emb.shape}) "
                                f"diferente do query ({query_embedding.shape}), ignorando.")
                continue
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-10
            )
            results.append((similarity, doc["text"]))
        results.sort(key=lambda x: x[0], reverse=True)
        logging.info(f"Top {k} documentos mais similares: {[r[0] for r in results[:k]]}")
        top_texts = [r[1] for r in results[:k]]
        return "\n".join(top_texts) if top_texts else "Nenhum contexto relevante encontrado."
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# -------- FUNÇÃO 3: gerar resposta com contexto --------
async def rag_answer(query: str):
    logging.info(f"Recebido query: {query}")
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
    logging.info(f"Prompt para LLM:\n{prompt}")

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
        answer = response.choices[0].message["content"]
        logging.info(f"Resposta gerada: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
