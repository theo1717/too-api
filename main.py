import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext

# RAG imports
from langchain.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import AIMessage, HumanMessage
from transformers import pipeline

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://toofrontend.vercel.app",
        "http://localhost:8081",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada no .env")

client = AsyncIOMotorClient(MONGO_URI)
db_users = client.users
collection_users = db_users.get_collection("too_users")

db_embeddings = client.file_data
collection_embeddings = db_embeddings.get_collection("embeddings")

# Segurança
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "CHAVE_MUITO_SECRETA"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# --- Pydantic Models ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = None

class LoginRequest(BaseModel):
    username: EmailStr
    password: str

class ChatRequest(BaseModel):
    chat_id: str | None = None
    message: str

# --- Funções auxiliares ---
def hash_password(password):
    return pwd_context.hash(password)

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(data: dict):
    data = data.copy()
    data["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# Dependência para obter usuário do token
from fastapi.security import OAuth2PasswordBearer
oauth2 = OAuth2PasswordBearer(tokenUrl="/login")

async def get_user_from_token(token: str = Depends(oauth2)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")

        if not email:
            raise HTTPException(status_code=401, detail="Token inválido")

        user = await collection_users.find_one({"email": email})

        if not user:
            raise HTTPException(status_code=401, detail="Usuário não encontrado")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")

# --- Inicializa LLM + embeddings RAG ---
llm_model_id = os.getenv("LLM_MODEL_ID", "decapoda-research/llama-7b-hf")  # exemplo
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

hf_pipe = pipeline("text-generation", model=llm_model_id, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# Carregar FAISS index local ou criar se não existir
if os.path.exists("index_faiss"):
    vectorstore = FAISS.load_local("index_faiss", HuggingFaceEmbeddings(model_name=embedding_model_name))
else:
    vectorstore = None

retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3, "fetch_k":4}) if vectorstore else None

chat_histories = {}  # Em memória. Pode migrar para MongoDB se quiser persistência.

# --- Funções RAG ---
def config_rag_chain(llm, retriever):
    system_prompt = "Você é Too, assistente virtual da Tecnotooling. Responda em português, claro e objetivo."
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}\n\nContexto: {context}")
    ])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
    return qa_chain

def chat_iteration(rag_chain, user_input, messages):
    messages.append(HumanMessage(content=user_input))
    resp = rag_chain.run({"query": user_input})
    messages.append(AIMessage(content=resp))
    return resp

# --- Rotas ---
@app.get("/")
async def root():
    return {"mensagem": "API funcionando!"}

# REGISTER
@app.post("/register")
async def register_user(user: UserRegister):
    existing = await collection_users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email já registrado")

    hashed = hash_password(user.password)
    new_user = {"email": user.email, "password": hashed}

    result = await collection_users.insert_one(new_user)
    return {"mensagem": "Usuário registrado", "user_id": str(result.inserted_id)}

# LOGIN
@app.post("/login")
async def login(data: LoginRequest):
    username = data.username
    password = data.password

    user = await collection_users.find_one({"email": username})

    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=400, detail="Email ou senha incorretos.")

    token = create_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

# LIST USERS
@app.get("/users")
async def list_users(current_user=Depends(get_user_from_token)):
    users = []
    async for u in collection_users.find():
        users.append({"email": u["email"], "id": str(u["_id"])})

    return users

# VALIDATE EMAIL
@app.post("/validate-email")
async def validate_email(email: EmailStr):
    exists = await collection_users.find_one({"email": email})
    return {"exists": bool(exists)}

# DELETE ACCOUNT
@app.delete("/delete-account")
async def delete_account(current_user=Depends(get_user_from_token)):
    await collection_users.delete_one({"email": current_user["email"]})
    return {"mensagem": "Conta deletada"}

# UPDATE ACCOUNT
@app.put("/update-account")
async def update_account(data: UserUpdate, current_user=Depends(get_user_from_token)):
    update_data = {}

    if data.email:
        update_data["email"] = data.email
    if data.password:
        update_data["password"] = hash_password(data.password)

    if not update_data:
        raise HTTPException(status_code=400, detail="Nenhum campo enviado")

    await collection_users.update_one(
        {"email": current_user["email"]},
        {"$set": update_data}
    )

    return {"mensagem": "Conta atualizada"}

# --- ROTA DE CHAT RAG ---
@app.post("/chat")
async def chat(req: ChatRequest, current_user=Depends(get_user_from_token)):
    if not retriever:
        raise HTTPException(status_code=500, detail="RAG não inicializado")
    
    chat_id = req.chat_id or str(datetime.utcnow().timestamp())
    if chat_id not in chat_histories:
        chat_histories[chat_id] = {"messages": []}

    messages = chat_histories[chat_id]["messages"]
    rag_chain = config_rag_chain(llm, retriever)
    answer = chat_iteration(rag_chain, req.message, messages)
    return {"answer": answer, "chat_id": chat_id}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
