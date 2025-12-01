from fastapi import HTTPException
from database.mongo import collection_users
from models.user_model import UserUpdate
from src.middlewares.security import hash_password

class UsuariosController:

    async def listar(self):
        users = collection_users.find({}, {"password": 0})
        return [u async for u in users]

    async def buscar_por_id(self, id: str):
        user = await collection_users.find_one({"_id": id}, {"password": 0})
        if not user:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        return user

    async def atualizar(self, id: str, data: UserUpdate):
        update_data = {}

        if data.email:
            update_data["email"] = data.email
        if data.password:
            update_data["password"] = hash_password(data.password)

        await collection_users.update_one({"_id": id}, {"$set": update_data})
        return {"mensagem": "Usuário atualizado"}

    async def deletar(self, id: str):
        result = await collection_users.delete_one({"_id": id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Usuário não encontrado")
        return {"mensagem": "Usuário removido"}

usuarios_controller = UsuariosController()
