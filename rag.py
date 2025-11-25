# rag.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class RAGPipeline:
    def __init__(self):
        self.embedder = None
        self.vectorstore = None
        self.tokenizer = None
        self.lm_model = None

    def load(self):
        print("\n🔄 Carregando modelos de IA (RAG)...")

        # Embeddings
        self.embedder = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Cria índice FAISS com texto inicial
        self.vectorstore = FAISS.from_texts(
            ["Sistema iniciado."],
            self.embedder
        )

        # Modelo da Google (Gemma)
        model_name = "google/gemma-2-2b-it"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.lm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("✅ Modelos carregados com sucesso.\n")

    def ask(self, query: str):
        # 1. Busca semântica
        docs = self.vectorstore.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])

        # 2. Prompt
        prompt = f"Contexto:\n{context}\n\nPergunta: {query}\nResposta:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.lm_model.device)

        output = self.lm_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.2
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# Instância global usada pelo FastAPI
rag_pipeline = RAGPipeline()
