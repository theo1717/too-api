# rag.py
import os
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import logging
import asyncio
import cohere

load_dotenv()

# ---- CONFIG ----
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not COHERE_API_KEY:
    raise Exception("Erro: COHERE_API_KEY não encontrada.")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

EMB_DIM = 384  # dimensao do embedding
TOP_K = 3      # número de docs para RAG

# ---- CLIENTE COHERE ----
co = cohere.Client(COHERE_API_KEY)

# ---- MONGO ----
client = AsyncIOMotorClient(MONGO_URI)
db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# ---- FUNÇÃO 1: gerar embedding via Cohere ----
async def get_embedding(text: str) -> np.ndarray:
    try:
        response = co.embed(
            model="small",
            texts=[text]
        )
        emb = np.array(response.embeddings[0])
        return emb
    except Exception as e:
        logging.error(f"Erro ao gerar embedding Cohere: {e}")
        return np.zeros(EMB_DIM)

# ---- FUNÇÃO 2: buscar top-k documentos similares ----
async def search_similar_docs(query_embedding, k=TOP_K):
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

# ---- FUNÇÃO 3: gerar resposta com RAG ----
async def rag_answer(query: str):
    # 1) gerar embedding da query
    query_emb = await get_embedding(query)
    
    # 2) buscar contexto relevante
    context = await search_similar_docs(query_emb)

    # 3) criar prompt
    prompt = f"""
Você é o assistente virtual da empresa TecnoTooling chamado "Too". 
Seu objetivo é fornecer respostas objetivas e claras usando o CONTEXTO RELEVANTE abaixo.

CONTEXTO RELEVANTE:
{context}

PERGUNTA:
{query}

Responda de forma direta, profissional e concisa, sem incluir informações irrelevantes.
"""
    logging.info(f"Prompt gerado:\n{prompt}")

    # 4) gerar resposta com Cohere (opcional, se quiser usar o endpoint generate)
    try:
        response = co.generate(
            model="xlarge",
            prompt=prompt,
            max_tokens=300,
            temperature=0.3
        )
        return response.generations[0].text.strip()
    except Exception as e:
        logging.error(f"Erro ao gerar resposta Cohere: {e}")
        return "Desculpe, não consegui gerar a resposta no momento."
