import os
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://toofrontend.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada no arquivo .env")

client = AsyncIOMotorClient(MONGO_URI)
db = client.too
collection_users = db.get_collection("too")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/login")

SECRET_KEY = "CHAVE_MUITO_SECRETA"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = None

# Funções auxiliares

def hash_password(password):
    return pwd_context.hash(password)


def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)


def create_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


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

# Endpoints

@app.get("/")
async def read_root():
    return {"Mensagem": "API de Onboarding no ar!"}

# Registro

@app.post("/register")
async def register_user(user: UserRegister):
    existing_user = await collection_users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Este email já está registado.")

    user.password = hash_password(user.password)

    new_user = await collection_users.insert_one(user.model_dump())

    return {
        "mensagem": "Utilizador registado com sucesso!",
        "user_id": str(new_user.inserted_id)
    }

# Login

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await collection_users.find_one({"email": form_data.username})
    if not user:
        raise HTTPException(status_code=400, detail="Email ou senha incorretos.")

    if not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=400, detail="Email ou senha incorretos.")

    token = create_token({"sub": user["email"]})

    return {"access_token": token, "token_type": "bearer"}

# Usuários

@app.get("/users")
async def list_users(current_user=Depends(get_user_from_token)):

    users = []
    async for user in collection_users.find():
        users.append({
            "email": user["email"],
            "_id": str(user["_id"])
        })

    return users

# Validar Emails

@app.post("/validate-email")
async def validate_email(email: EmailStr):
    exists = await collection_users.find_one({"email": email})
    return {"exists": bool(exists)}

# Logout

@app.post("/logout")
async def logout():
    # Em JWT não existe logout real → o frontend apenas apaga o token
    return {"mensagem": "Logout efetuado. Apague o token no frontend."}


# Deletar Email

@app.delete("/delete-account")
async def delete_account(current_user=Depends(get_user_from_token)):
    await collection_users.delete_one({"email": current_user["email"]})
    return {"mensagem": "Conta apagada com sucesso."}


# Atualizar contas

@app.put("/update-account")
async def update_account(data: UserUpdate, current_user=Depends(get_user_from_token)):

    update_data = {}

    if data.email:
        update_data["email"] = data.email

    if data.password:
        update_data["password"] = hash_password(data.password)

    if not update_data:
        raise HTTPException(status_code=400, detail="Nenhum campo enviado.")

    await collection_users.update_one(
        {"email": current_user["email"]},
        {"$set": update_data}
    )

    return {"mensagem": "Conta atualizada com sucesso."}

# rodar localmente

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
