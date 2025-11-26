# rag.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import cohere
import asyncio
import groq  # Groq SDK para chat generation

load_dotenv()

# ---- CONFIGURAÇÕES COHERE ----
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise Exception("Erro: COHERE_API_KEY não encontrada.")
co = cohere.Client(COHERE_API_KEY)

EMB_DIM = 384  # dimensão do embedding small
TOP_K = 3      # quantos trechos retornar

# ---- CONFIGURAÇÕES GROQ ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise Exception("Erro: GROQ_API_KEY não encontrada.")
groq_client = groq.Client(api_key=GROQ_API_KEY)

# ---- MONGO ----
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# ---- FUNÇÃO 1: gerar embedding via Cohere ----
async def get_embedding(text: str) -> np.ndarray:
    try:
        response = co.embed(model="small", texts=[text])
        emb = np.array(response.embeddings[0])
        if emb.shape[0] != EMB_DIM:
            logging.warning(f"Embedding retornou {emb.shape[0]} dimensões, esperado {EMB_DIM}")
        return emb
    except Exception as e:
        logging.error(f"Erro ao gerar embedding Cohere: {e}")
        return np.zeros(EMB_DIM)

# ---- FUNÇÃO 2: buscar top-k documentos similares ----
async def search_similar_docs(query_embedding: np.ndarray, k: int = TOP_K) -> str:
    results = []
    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc.get("embedding", np.zeros(EMB_DIM)))
            if doc_emb.shape != query_embedding.shape:
                logging.warning(f"Dimensões diferentes: doc {doc_emb.shape}, query {query_embedding.shape}")
                continue
            similarity = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-10
            )
            results.append((similarity, doc.get("text", "")))

        results.sort(key=lambda x: x[0], reverse=True)
        top_texts = [r[1] for r in results[:k]]
        logging.info(f"Top {k} documentos mais similares: {[r[0] for r in results[:k]]}")
        return "\n".join(top_texts)
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# ---- FUNÇÃO 3: gerar resposta com RAG via Groq ----
async def rag_answer(query: str, chat_id: str = None, user_email: str = None) -> str:
    """
    Gera resposta de chat usando:
        - Cohere embeddings para buscar contexto
        - Groq para gerar resposta baseada no contexto
    """
    # 1️⃣ Buscar contexto
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    # 2️⃣ Criar prompt
    prompt = f"""
Você é o assistente virtual da empresa TecnoTooling chamado "Too". 
Seu objetivo é fornecer respostas objetivas e claras usando o CONTEXTO RELEVANTE abaixo.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma direta, profissional e concisa, sem incluir informações irrelevantes.
"""

    logging.info(f"Prompt enviado para Groq:\n{prompt}")

    # 3️⃣ Chamar Groq Chat
    try:
        response = groq_client.chat(
            model="gpt-4.1-mini",  # ou outro modelo disponível
            messages=[
                {"role": "system", "content": "Você é um assistente profissional."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.message.content.strip()
    except Exception as e:
        logging.error(f"Erro ao gerar resposta Groq: {e}")
        return "Desculpe, não consegui gerar uma resposta no momento."
