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
HF_MODEL = "BAAI/bge-m3-small"   # Modelo Hugging Face embeddings

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
            if len(emb.shape) > 1:  # média dos tokens
                emb = emb.mean(axis=0)
            return emb
        except Exception as e:
            logging.error(f"Erro ao gerar embedding HF: {e}")
            return np.zeros(1024)  # dimensão do BGE-M3-Small

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
        logging.info(f"Top {k} documentos mais similares: {[r[0] for r in results[:k]]}")
        return "\n".join(top_texts)
    except Exception as e:
        logging.error(f"Erro ao buscar documentos similares: {e}")
        return "Não foi possível buscar contexto relevante."

# ---- FUNÇÃO 3: construir prompt amigável ----
def build_too_prompt(user_message: str, context: str) -> str:
    return f"""
Você é o Too, um assistente virtual da TecnoTooling super amigável, querido e receptivo.
Sempre responda de forma acolhedora, clara e gentil, como se estivesse ajudando um amigo.
Use o CONTEXTO RELEVANTE abaixo para responder de forma precisa.

CONTEXTO RELEVANTE:
{context}

PERGUNTA DO USUÁRIO:
{user_message}

Responda como o Too, sendo útil, simpático e empático.
"""

# ---- FUNÇÃO 4: gerar resposta com RAG via Groq ----
async def rag_answer(query: str, chat_id: str = None, user_email: str = None):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    # Verifica se é a primeira mensagem do chat
    saudacao_inicial = ""
    if chat_id and user_email:
        previous_msgs = collection_embeddings.count_documents({"chat_id": chat_id})
        if previous_msgs == 0:
            saudacao_inicial = "Oi! Tudo bem? 😊 "

    prompt = f"""
{saudacao_inicial}
Você é o Too, um assistente virtual da TecnoTooling super amigável, querido e receptivo.
Sempre responda de forma acolhedora, clara e gentil, como se estivesse ajudando um amigo.
Use o CONTEXTO RELEVANTE abaixo para responder de forma precisa.

CONTEXTO RELEVANTE:
{context}

PERGUNTA DO USUÁRIO:
{query}

Responda como o Too, sendo útil, simpático e empático.
"""

    logging.info(f"Prompt para LLM:\n{prompt}")

    if not groq_client:
        logging.warning("Groq client não disponível, retornando fallback")
        return "Desculpe, não consigo gerar resposta no momento."

    try:
        response = groq_client.chat.completions.create(
            model="groq/compound-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente útil e amigável."},
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
        logging.error(f"Erro ao gerar resposta: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
