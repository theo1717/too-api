from fastapi import APIRouter
from controllers.auth_controller import auth_controller
from models.user_model import UserRegister, LoginRequest

router = APIRouter()

# POST /api/usuarios/login
@router.post("/login")
async def login(data: LoginRequest):
    return await auth_controller.login(data)

# POST /api/usuarios (cria usuário)
@router.post("/")
async def criar_usuario(data: UserRegister):
    return await auth_controller.registrar(data)
