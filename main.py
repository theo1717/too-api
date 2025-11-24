import os
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends, Form
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext

load_dotenv()

app = FastAPI()

# -----------------------------------------
# CORS
# -----------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # liberar geral pois é mobile + web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# MONGO
# -----------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada")

client = AsyncIOMotorClient(MONGO_URI)
db = client.users
collection_users = db.too_users  # nome que você usa no login

# -----------------------------------------
# AUTENTICAÇÃO
# -----------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/login")

SECRET_KEY = "CHAVE_MUITO_SECRETA"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MIN = 60


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)


def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MIN)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_user_from_token(token: str = Depends(oauth2)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")

        if not email:
            raise HTTPException(401, "Token inválido")

        user = await collection_users.find_one({"email": email})
        if not user:
            raise HTTPException(401, "Usuário não encontrado")

        return user

    except JWTError:
        raise HTTPException(401, "Token inválido")

# -----------------------------------------
# MODELOS
# -----------------------------------------
class UserRegister(BaseModel):
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    password: str | None = None


# -----------------------------------------
# ROTAS
# -----------------------------------------
@app.get("/")
async def home():
    return {"mensagem": "API funcionando!"}


# -------- REGISTER --------
@app.post("/register")
async def register_user(user: UserRegister):

    exists = await collection_users.find_one({"email": user.email})
    if exists:
        raise HTTPException(400, "Email já cadastrado")

    hashed = hash_password(user.password)

    result = await collection_users.insert_one({
        "email": user.email,
        "password": hashed,
    })

    return {
        "mensagem": "Usuário cadastrado com sucesso!",
        "user_id": str(result.inserted_id)
    }


# -------- LOGIN (FormData) --------
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """
    O front envia:
    formData.append("username", email)
    formData.append("password", password)
    """

    user = await collection_users.find_one({"email": username})

    if not user:
        raise HTTPException(400, "Email ou senha incorretos")

    if not verify_password(password, user["password"]):
        raise HTTPException(400, "Email ou senha incorretos")

    token = create_token({"sub": user["email"]})

    return {"access_token": token, "token_type": "bearer"}


# -------- LIST USERS --------
@app.get("/users")
async def list_users(current_user=Depends(get_user_from_token)):
    users = []

    async for user in collection_users.find():
        users.append({
            "_id": str(user["_id"]),
            "email": user["email"]
        })

    return users


# -------- VALIDATE EMAIL --------
@app.post("/validate-email")
async def validate_email(email: EmailStr):
    exists = await collection_users.find_one({"email": email})
    return {"exists": bool(exists)}


# -------- LOGOUT --------
@app.post("/logout")
async def logout():
    return {"mensagem": "Logout efetuado. Apague o token no frontend."}


# -------- DELETE ACCOUNT --------
@app.delete("/delete-account")
async def delete_account(current_user=Depends(get_user_from_token)):
    await collection_users.delete_one({"email": current_user["email"]})
    return {"mensagem": "Conta deletada com sucesso."}


# -------- UPDATE ACCOUNT --------
@app.put("/update-account")
async def update_account(data: UserUpdate, current_user=Depends(get_user_from_token)):

    update_data = {}

    if data.email:
        update_data["email"] = data.email

    if data.password:
        update_data["password"] = hash_password(data.password)

    if not update_data:
        raise HTTPException(400, "Nenhum campo para atualizar")

    await collection_users.update_one(
        {"email": current_user["email"]},
        {"$set": update_data}
    )

    return {"mensagem": "Conta atualizada com sucesso."}


# -----------------------------------------
# RODAR LOCAL
# -----------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
