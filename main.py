from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.auth_routes import router as auth_routes
from routes.chat_routes import router as chat_routes
from routes.usuarios_routes import router as usuarios_routes

app = FastAPI(title="Too AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes, prefix="/auth")
app.include_router(chat_routes, prefix="/chat")
app.include_router(usuarios_routes, prefix="/usuarios")

@app.get("/")
def root():
    return {"status": "API rodando com RAG Cohere + Groq"}
