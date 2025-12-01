from pydantic import BaseModel

class ChatMessage(BaseModel):
    chat_id: str | None = None
    user_email: str
    message: str

