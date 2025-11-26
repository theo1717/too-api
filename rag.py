# rag.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import logging

load_dotenv()

# ---- CONFIG ----
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")  # Token Hugging Face
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Modelo gratuito Hugging Face

if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")
if not HF_TOKEN:
    raise Exception("Erro: HF_TOKEN não encontrada.")

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# ---- FUNÇÃO 1: gerar embedding via Hugging Face (async) ----
async def get_embedding(text: str) -> np.ndarray:
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text}

    async with httpx.AsyncClient(timeout=60) as session:
        try:
            response = await session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            emb = np.array(response.json())
            if len(emb.shape) > 1:  # média dos tokens
                emb = emb.mean(axis=0)
            return emb
        except Exception as e:
            logging.error(f"Erro ao gerar embedding HF: {e}")
            return np.zeros(384)  # dimensão do MiniLM

# ---- FUNÇÃO 2: buscar top-k documentos similares ----
async def search_similar_docs(query_embedding, k=3):
    results = []
    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc["embedding"])
            q_emb = np.array(query_embedding)
            if doc_emb.shape != q_emb.shape:
                logging.warning(f"Dimensões diferentes: doc {doc_emb.shape}, query {q_emb.shape}")
                continue
            similarity = np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb) + 1e-10)
            results.append((similarity, doc["text"]))

        results.sort(key=lambda x: x[0], reverse=True)
        top_texts = [r[1] for r in results[:k]]
        return "\n".join(top_texts)
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return ""

# ---- FUNÇÃO 3: construir prompt objetivo ----
def build_too_prompt(user_message: str, context: str) -> str:
    return f"""
Você é o Too, assistente virtual da TecnoTooling.
Seja objetivo, conciso e útil, usando apenas o CONTEXTO RELEVANTE abaixo.

CONTEXTO RELEVANTE:
{context}

PERGUNTA DO USUÁRIO:
{user_message}

Responda de forma direta e clara, sem cumprimentos desnecessários.
"""

# ---- FUNÇÃO 4: gerar resposta com RAG ----
async def rag_answer(query: str, chat_id: str = None, user_email: str = None):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    # ---- Saudação inicial apenas se primeira mensagem do chat ----
    saudacao_inicial = ""
    if chat_id and user_email:
        previous_msgs_count = await collection_embeddings.count_documents({"chat_id": chat_id})
        if previous_msgs_count == 0:
            saudacao_inicial = "Oi! 😊 Aqui é o Too, seu assistente. "

    prompt = saudacao_inicial + build_too_prompt(query, context)
    logging.info(f"Prompt gerado:\n{prompt}")

    # ---- Chamando LLM (exemplo Groq ou outro) ----
    # Aqui colocamos apenas fallback de teste
    response_text = prompt  # substitua pela chamada real do LLM se desejar

    return response_text
