# main.py

import os
import uvicorn
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# --- Configuração do Banco de Dados ---

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada no ficheiro .env")

client = AsyncIOMotorClient(MONGO_URI)

db = client.too

collection_users = db.get_collection("too")


# --- Modelos de Dados (Pydantic) ---

class UserRegister(BaseModel):
    email: EmailStr
    password: str

class UserInDB(BaseModel):
    email: EmailStr

app = FastAPI()


# --- Endpoints

@app.get("/")
async def read_root():
    return {"Mensagem": "API de Onboarding no ar!"}


# Endpoint 2: Registar Utilizador
@app.post("/register")
async def register_user(user: UserRegister):
    
    existing_user = await collection_users.find_one({"email": user.email})
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este email já está registado.",
        )

    user_data = user.model_dump()
    

    # Insere o novo utilizador no banco de dados
    new_user = await collection_users.insert_one(user_data)

    # Retorna uma mensagem de sucesso
    return {
        "mensagem": "Utilizador registado com sucesso!",
        "user_id": str(new_user.inserted_id)
    }

# --- rodar localmente ---

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)