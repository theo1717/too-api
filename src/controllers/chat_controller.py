from fastapi import APIRouter, Depends
from datetime import datetime
from app.models.chat_model import ChatRequest
from app.database.mongo import collection_history
from app.middlewares.security import get_user_from_token
from app.services.rag_service import rag_answer
from pydantic import BaseModel
from fastapi import HTTPException
from services.rag_service import run_rag

router = APIRouter()

@router.post("/chat")
async def chat(req: ChatRequest, current_user=Depends(get_user_from_token)):

    chat_id = req.chat_id or str(datetime.utcnow().timestamp())

    await collection_history.insert_one({
        "chat_id": chat_id,
        "user_email": current_user["email"],
        "sender": "user",
        "text": req.message,
        "timestamp": datetime.utcnow()
    })

    answer = await rag_answer(
        query=req.message,
        chat_id=chat_id,
        user_email=current_user["email"]
    )

    await collection_history.insert_one({
        "chat_id": chat_id,
        "user_email": current_user["email"],
        "sender": "bot",
        "text": answer,
        "timestamp": datetime.utcnow()
    })

    return {"answer": answer, "chat_id": chat_id}
