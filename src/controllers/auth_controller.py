from fastapi import HTTPException
from models.user_model import UserRegister, LoginRequest
from database.mongo import collection_users
from middlewares.security import hash_password, verify_password, create_token

class AuthController:

    async def login(self, data: LoginRequest):
        user = await collection_users.find_one({"email": data.username})
        if not user or not verify_password(data.password, user["password"]):
            raise HTTPException(status_code=400, detail="Email ou senha incorretos.")

        token = create_token({"sub": user["email"], "role": user.get("role", "user")})
        return {"access_token": token, "token_type": "bearer"}

    async def registrar(self, user: UserRegister):
        existing = await collection_users.find_one({"email": user.email})
        if existing:
            raise HTTPException(status_code=400, detail="Email já registrado")

        hashed = hash_password(user.password)

        result = await collection_users.insert_one({
            "email": user.email,
            "password": hashed,
            "is_admin": False
        })

        return {"mensagem": "Usuário registrado", "user_id": str(result.inserted_id)}

auth_controller = AuthController()
