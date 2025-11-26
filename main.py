# main.py
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
from fastapi.security import OAuth2PasswordBearer
from pymongo import MongoClient

# --- RAG Import ---
from rag import rag_answer, get_embedding  # usamos o HF embeddings aqui

load_dotenv()

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MONGO ---
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

client = AsyncIOMotorClient(MONGO_URI)
db_users = client.users
collection_users = db_users.get_collection("too_users")
db_chats = client.chats
collection_history = db_chats.get_collection("history")

# --- SEGURANÇA ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "CHAVE_MUITO_SECRETA")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2 = OAuth2PasswordBearer(tokenUrl="/login")

# --- MODELS ---
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

# --- FUNÇÕES AUXILIARES ---
def hash_password(password):
    return pwd_context.hash(password[:72])

def verify_password(password, hashed):
    return pwd_context.verify(password[:72], hashed)

def create_token(data: dict):
    data = data.copy()
    data["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

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

# --- ROTAS ---
@app.get("/")
async def root():
    return {"mensagem": "API funcionando!"}

@app.post("/register")
async def register_user(user: UserRegister):
    existing = await collection_users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email já registrado")
    hashed = hash_password(user.password)
    new_user = {"email": user.email, "password": hashed}
    result = await collection_users.insert_one(new_user)
    return {"mensagem": "Usuário registrado", "user_id": str(result.inserted_id)}

@app.post("/login")
async def login(data: LoginRequest):
    user = await collection_users.find_one({"email": data.username})
    if not user or not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Email ou senha incorretos.")
    token = create_token({"sub": user["email"]})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat")
async def chat(req: ChatRequest, current_user=Depends(get_user_from_token)):
    chat_id = req.chat_id or str(datetime.utcnow().timestamp())

    # --- Salva a mensagem do usuário ---
    user_msg = {
        "chat_id": chat_id,
        "user_email": current_user["email"],
        "sender": "user",
        "text": req.message,
        "timestamp": datetime.utcnow()
    }
    await collection_history.insert_one(user_msg)

    # --- Gera resposta com RAG ---
    answer_text = await rag_answer(req.message)

    # --- Salva a resposta do bot ---
    bot_msg = {
        "chat_id": chat_id,
        "user_email": current_user["email"],
        "sender": "bot",
        "text": answer_text,
        "timestamp": datetime.utcnow()
    }
    await collection_history.insert_one(bot_msg)

    # --- Retorna resposta e chat_id ---
    return {"answer": answer_text, "chat_id": chat_id}

@app.get("/chat-history")
async def chat_list(current_user=Depends(get_user_from_token)):
    # Agrupa mensagens por chat_id
    pipeline = [
        {"$match": {"user_email": current_user["email"]}},
        {"$sort": {"timestamp": 1}},
        {
            "$group": {
                "_id": "$chat_id",
                "first_message": {"$first": "$text"},
                "date": {"$first": "$timestamp"}
            }
        },
        {"$sort": {"date": -1}}
    ]

    cursor = collection_history.aggregate(pipeline)

    chats = []
    async for chat in cursor:
        chats.append({
            "id": chat["_id"],
            "title": chat["first_message"][:40] + "...",
            "date": chat["date"].strftime("%Y-%m-%d"),
        })

    return {"chats": chats}

from fastapi import Body

# ------------------------------
# 1) CRIAR NOVO CHAT
# ------------------------------
@app.post("/new-chat")
async def new_chat(current_user=Depends(get_user_from_token)):
    new_id = str(datetime.utcnow().timestamp())

    # salva um registro inicial (vazio)
    await collection_history.insert_one({
        "chat_id": new_id,
        "user_email": current_user["email"],
        "sender": "system",
        "text": "Chat criado",
        "timestamp": datetime.utcnow()
    })

    return {"chat_id": new_id}


# ------------------------------
# 2) PEGAR MENSAGENS DE UM CHAT ESPECÍFICO
# ------------------------------
@app.get("/chat-history/{chat_id}")
async def get_chat_messages(chat_id: str, current_user=Depends(get_user_from_token)):

    cursor = collection_history.find(
        {"chat_id": chat_id, "user_email": current_user["email"]},
        sort=[("timestamp", 1)]
    )

    msgs = []
    async for m in cursor:
        msgs.append({
            "sender": m.get("sender"),
            "text": m.get("text"),
            "timestamp": m.get("timestamp")
        })

    return {"chat_id": chat_id, "messages": msgs}


# ------------------------------
# 3) ENVIAR MENSAGEM (CASO SEU FRONT PRECISE)
# MAS O SEU FRONT JÁ USA /chat, ENTÃO É OPCIONAL
# ------------------------------
@app.post("/send-message")
async def send_message(
    data: dict = Body(...),
    current_user=Depends(get_user_from_token)
):
    chat_id = data.get("chat_id")
    text = data.get("text")

    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id ausente")

    # salva msg do usuário
    await collection_history.insert_one({
        "chat_id": chat_id,
        "user_email": current_user["email"],
        "sender": "user",
        "text": text,
        "timestamp": datetime.utcnow()
    })

    # gera resposta via RAG
    answer_text = await rag_answer(text)

    # salva msg do bot
    await collection_history.insert_one({
        "chat_id": chat_id,
        "user_email": current_user["email"],
        "sender": "bot",
        "text": answer_text,
        "timestamp": datetime.utcnow()
    })

    return {"answer": answer_text}

from fastapi import Path

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str = Path(...), current_user=Depends(get_user_from_token)):
    result = await collection_history.delete_many({"chat_id": chat_id, "user_email": current_user["email"]})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat não encontrado")
    return {"mensagem": f"Chat {chat_id} deletado com sucesso."}
