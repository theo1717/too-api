# rag.py
import os
import numpy as np
from groq import Groq
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import httpx
import logging
import asyncio

load_dotenv()

# ---- CONFIG ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
HF_TOKEN = os.getenv("HF_TOKEN")  # Token Hugging Face
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # embedding de 384
MB_DIM = 384  # dimensão correta do BGE-M3-Small

if not GROQ_API_KEY:
    raise Exception("Erro: GROQ_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")
if not HF_TOKEN:
    raise Exception("Erro: HF_TOKEN não encontrada.")

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
            if len(emb.shape) > 1:
                emb = emb.mean(axis=0)  # média dos tokens
            return emb
        except Exception as e:
            logging.error(f"Erro ao gerar embedding HF: {e}")
            return np.zeros(EMB_DIM)

# ---- FUNÇÃO 2: buscar top-k documentos similares ----
async def search_similar_docs(query_embedding, k=3):
    results = []
    try:
        async for doc in collection_embeddings.find():
            doc_emb = np.array(doc["embedding"])
            if doc_emb.shape != query_embedding.shape:
                logging.warning(f"Dimensões diferentes: doc {doc_emb.shape}, query {query_embedding.shape}")
                continue
            similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb) + 1e-10)
            results.append((similarity, doc["text"]))

        results.sort(key=lambda x: x[0], reverse=True)
        top_texts = [r[1] for r in results[:k]]
        logging.info(f"Top {k} documentos mais similares: {[r[0] for r in results[:k]]}")
        return "\n".join(top_texts)
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# ---- FUNÇÃO 3: gerar resposta com RAG via Groq ----
async def rag_answer(query: str, chat_id: str | None = None, user_email: str | None = None) -> str:
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    # Saudação ou contexto adicional se for primeira mensagem
    greeting = f"Usuário: {user_email}\n" if user_email else ""

    prompt = f"""
Você é o assistente virtual da empresa TecnoTooling chamado "Too". 
Seu objetivo é fornecer respostas objetivas e claras usando o CONTEXTO RELEVANTE abaixo.

CONTEXTO RELEVANTE:
{context}

{greeting}PERGUNTA:
{query}

Responda de forma direta, profissional e concisa, sem incluir informações irrelevantes.
"""
    logging.info(f"Prompt para LLM:\n{prompt}")

    if not groq_client:
        logging.warning("Groq client não disponível, retornando fallback")
        return "Desculpe, não consigo gerar resposta no momento."

    try:
        response = groq_client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        message_obj = response.choices[0].message
        if hasattr(message_obj, "content"):
            return message_obj.content
        elif isinstance(message_obj, list):
            return " ".join([m.content for m in message_obj])
        else:
            return str(message_obj)
    except Exception as e:
        logging.error(f"Erro ao gerar resposta LLM: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."

# ---- FUNÇÃO EXTRA: atualizar embeddings antigos ----
async def update_all_embeddings():
    async for doc in collection_embeddings.find():
        text = doc["text"]
        emb = await get_embedding(text)
        await collection_embeddings.update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": emb.tolist()}}
        )
    logging.info("Todos os embeddings atualizados para dimensão correta.")
