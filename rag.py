# rag.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import asyncio
from openai import OpenAI

load_dotenv()

# ---- CONFIG ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not OPENAI_API_KEY:
    raise Exception("Erro: OPENAI_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

# ---- CLIENTE OPENAI ----
openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMB_DIM = 384  # dimensão do embedding que você quer usar

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# ---- FUNÇÃO 1: gerar embedding via OpenAI ----
async def get_embedding(text: str) -> np.ndarray:
    try:
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        emb = np.array(resp.data[0].embedding)
        if emb.shape[0] != EMB_DIM:
            logging.warning(f"Embedding retornado tem dimensão {emb.shape[0]}, esperada {EMB_DIM}")
            # Ajusta para 384 preenchendo com zeros ou truncando
            if emb.shape[0] > EMB_DIM:
                emb = emb[:EMB_DIM]
            else:
                emb = np.pad(emb, (0, EMB_DIM - emb.shape[0]))
        return emb
    except Exception as e:
        logging.error(f"Erro ao gerar embedding OpenAI: {e}")
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

# ---- FUNÇÃO 3: gerar resposta com RAG usando OpenAI Chat ----
async def rag_answer(query: str):
    query_emb = await get_embedding(query)
    context = await search_similar_docs(query_emb)

    prompt = f"""
Você é o assistente virtual da empresa TecnoTooling chamado "Too". 
Seu objetivo é fornecer respostas objetivas e claras usando o CONTEXTO RELEVANTE abaixo.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma direta, profissional e concisa, sem incluir informações irrelevantes.
"""
    logging.info(f"Prompt para OpenAI Chat:\n{prompt}")

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return resp.choices[0].message.content
    except Exception as e:
        logging.error(f"Erro ao gerar resposta OpenAI: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."
