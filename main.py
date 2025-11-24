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
db = client.users
collection_users = db.get_collection("too_users")

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


# LOGIN (JSON)
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
