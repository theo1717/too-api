from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient

# --- Inicializa LLM
def load_llm(model_id="deepseek-r1-distill-llama-70b", temperature=0.7):
    return ChatGroq(model=model_id, temperature=temperature)

# --- Inicializa embeddings do MongoDB
def init_embeddings():
    client = MongoClient("mongodb+srv://dev:admin@cluster0.o9emz.mongodb.net/")
    db = client.file_data
    collection = db.embeddings
    # Aqui você cria FAISS retriever a partir dos embeddings salvos
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = FAISS.load_local("index_faiss")  # Se já tiver index salvo
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3, "fetch_k":4})

# --- Configura chain RAG
def config_rag_chain(llm, retriever):
    system_prompt = "Você é Too, assistente virtual da Tecnotooling. Responda em português, claro e objetivo."
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\n Contexto: {context}")
    ])
    return retriever, qa_prompt

# --- Iteração de chat
def chat_iteration(rag_chain, user_input, messages):
    messages.append(HumanMessage(content=user_input))
    resp = rag_chain.invoke({"input": user_input, "chat_history": messages})
    answer = resp.get("answer", "")
    messages.append(AIMessage(content=answer))
    return answer
