import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Mongo ----
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.too
collection_users = db.get_collection("too")

# ---- Modelo ----
class UserLogin(BaseModel):
    email: EmailStr
    password: str

@app.get("/")
async def home():
    return {"mensagem": "API funcionando!"}

@app.post("/login")
async def login(user: UserLogin):

    user_db = await collection_users.find_one({"email": user.email})

    if not user_db:
        raise HTTPException(400, "Email ou senha incorretos")

    if user_db["password"] != user.password:
        raise HTTPException(400, "Email ou senha incorretos")

    return {"mensagem": "Login OK", "email": user.email}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
