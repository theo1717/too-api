# rag.py
import os
import fitz
import gc
import shutil
import tempfile
import stat

from langchain.chains import create_stuff_documents_chain, create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

from transformers import pipeline

# Configurações
FAISS_INDEX_PATH = "index_faiss"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")  # ou outro
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "google/flan-t5-base")  # exemplo HuggingFace

# Funções auxiliares
def _handle_remove_readonly(func, path, excinfo):
    try:
        os.chmod(path, stat.S_IWRITE)
    except:
        pass
    func(path)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text = ""
    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in pdf_doc:
            text += page.get_text("text")
        pdf_doc.close()
    except Exception as e:
        print(f"Erro extraindo PDF: {e}")
    return text

def create_retriever_from_texts(pdf_texts):
    if not pdf_texts:
        raise ValueError("Nenhum texto para indexar.")

    gc.collect()
    index_path_to_use = FAISS_INDEX_PATH
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH, onerror=_handle_remove_readonly)
    except PermissionError:
        fallback_name = FAISS_INDEX_PATH + "_old"
        if os.path.exists(fallback_name):
            shutil.rmtree(fallback_name, onerror=_handle_remove_readonly)
        os.rename(FAISS_INDEX_PATH, fallback_name)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for t in pdf_texts:
        chunks.extend(text_splitter.split_text(t))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    try:
        os.makedirs(index_path_to_use, exist_ok=True)
        vectorstore.save_local(index_path_to_use)
    except Exception:
        tmp_dir2 = tempfile.mkdtemp(prefix="index_faiss_fallback_")
        vectorstore.save_local(tmp_dir2)
        index_path_to_use = tmp_dir2

    info = {
        "num_pdfs": len(pdf_texts),
        "num_chunks": len(chunks),
        "embedding_dim": len(vectorstore.index.reconstruct(0)) if len(chunks) > 0 else 0
    }

    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3, "fetch_k":4}), info

# --- LLM ---
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        max_length=512
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

def config_rag_chain(retriever):
    # History-aware retriever
    context_q_system = ("Given the following chat history and the follow-up question which might "
                        "reference context in the chat history, formulate a standalone question "
                        "which can be understood without the chat history. Do NOT answer the question; "
                        "just reformulate if needed.")
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}")
    ])
    history_retriever = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=context_q_prompt)

    system_prompt = """Você é Too, assistente virtual da Tecnotooling. Use os documentos fornecidos para responder.
    Se não souber, diga que não tem certeza. Responda em português, claro e objetivo.\n\n"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\n Contexto: {context}")
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_retriever, qa_chain)

def chat_iteration(rag_chain, user_input, messages):
    messages.append(HumanMessage(content=user_input))
    resp = rag_chain.invoke({"input": user_input, "chat_history": messages})
    answer = resp.get("answer", "")
    answer = answer.split("</think>")[-1].strip() if "</think>" in answer else answer.strip()
    messages.append(AIMessage(content=answer))
    return answer, messages
