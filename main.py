import os
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MongoDB ---
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.users       # banco = "users"
collection_users = db.too_users   # coleção = "too_users"


@app.get("/")
async def home():
    return {"mensagem": "API funcionando!"}


# ----------------------------------------------------
# 🔐 LOGIN — RECEBE username e password via FORM-DATA
# ----------------------------------------------------
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    print("🔎 Recebido do front:", username, password)

    user_db = await collection_users.find_one({"email": username})

    if not user_db:
        raise HTTPException(400, "Email ou senha incorretos")

    # No Mongo Atlas o campo é "senha"
    if user_db["senha"] != password:
        raise HTTPException(400, "Email ou senha incorretos")

    return {"mensagem": "Login OK", "email": username}


# --- Rodar local ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
