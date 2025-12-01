# mongo.py
import os
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise Exception("Erro: MONGO_URI não encontrada.")

client = AsyncIOMotorClient(MONGO_URI)

db_users = client.users
collection_users = db_users.get_collection("too_users")

db_chats = client.chats
collection_history = db_chats.get_collection("history")
