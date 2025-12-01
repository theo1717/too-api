from fastapi import APIRouter, Depends
from models.chat_model import ChatMessage
from controllers.chat_controller import process_message
from middlewares.security import get_current_user
from src.controllers import chat_controller

router = APIRouter()

@router.post("/send")
async def send_message(data: ChatMessage, user=Depends(get_current_user)):
    return await process_message(data)


router = APIRouter()

@router.post("/chat")
async def chat_route(body: ChatRequest, user=Depends(get_user_from_token)):
    return await chat_controller(body, user["email"])
