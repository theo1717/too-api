# rag.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import logging
import asyncio

load_dotenv()

# ---- CONFIG ----
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face API token
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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
            # Reduz embeddings de tokens para um vetor único
            if len(emb.shape) > 1:
                emb = emb.mean(axis=0)
            return emb  # dimensão 384
        except Exception as e:
            logging.error(f"Erro ao gerar embedding HF: {e}")
            return np.zeros(384)

# ---- FUNÇÃO 2: buscar top-k documentos similares ----
async def search_similar_docs(query_embedding, k=3, min_similarity=0.6):
    results = []
    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc["embedding"])
            q_emb = np.array(query_embedding)
            if doc_emb.shape != q_emb.shape:
                logging.warning(f"Dimensões diferentes: doc {doc_emb.shape}, query {q_emb.shape}")
                continue
            similarity = np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb) + 1e-10)
            if similarity >= min_similarity:
                results.append((similarity, doc["text"][:500]))  # limita 500 chars
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

# ---- FUNÇÃO 4: gerar resposta com RAG via Hugging Face API ----
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    prompt = build_too_prompt(query, context)
    logging.info(f"Prompt gerado:\n{prompt}")

    # ---- Chamando LLM da Hugging Face (ou outro endpoint) ----
    HF_CHAT_MODEL = "gpt2"  # substitua por outro modelo se tiver disponível

    url = f"https://api-inference.huggingface.co/models/{HF_CHAT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}

    async with httpx.AsyncClient(timeout=60) as session:
        try:
            response = await session.post(url, headers=headers, json=payload)
            response.raise_for_status()
            output = response.json()
            # Hugging Face retorna lista de strings ou dicionário dependendo do modelo
            if isinstance(output, list) and "generated_text" in output[0]:
                return output[0]["generated_text"]
            elif isinstance(output, dict) and "generated_text" in output:
                return output["generated_text"]
            else:
                return str(output)
        except Exception as e:
            logging.error(f"Erro ao gerar resposta LLM: {e}")
            return "Desculpe, não consegui gerar a resposta."
