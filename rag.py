# rag.py
import os
from groq import Groq
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

GROQ_KEY = os.getenv("GROQ_API_KEY")

class RAGPipeline:
    def __init__(self):
        self.embedder = None
        self.vectorstore = None
        self.client = None
        self.model_name = "gemma2-9b-it"

    def load(self):
        print("\n🔄 Carregando embeddings + Groq...")

        # Embeddings locais (muito leves)
        self.embedder = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vetor FAISS inicial
        self.vectorstore = FAISS.from_texts(
            ["Sistema iniciado."],
            self.embedder
        )

        # Cliente Groq
        self.client = Groq(api_key=GROQ_KEY)

        print("✅ RAG carregado com Groq.\n")

    def ask(self, query: str):
        # 1. Busca semântica
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])

        # 2. Prompt final
        prompt = f"""
Use o contexto abaixo para responder a pergunta.

Contexto:
{context}

Pergunta:
{query}

Resposta:
"""

        # 3. Chamada ao Groq
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[ {"role": "user", "content": prompt} ],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message["content"]


# Instância global
rag_pipeline = RAGPipeline()

def load_vectorstore():
    rag_pipeline.load()
    return rag_pipeline.vectorstore

def config_rag_chain(_=None):
    return rag_pipeline

def chat_iteration(rag_chain, message, _messages):
    return rag_chain.ask(message)
