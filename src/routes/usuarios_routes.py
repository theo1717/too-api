from fastapi import APIRouter, Depends
from controllers.usuarios_controller import usuarios_controller
from middlewares.security import get_user_from_token
from middlewares.roles import verificaAdmin
from models.user_model import UserUpdate

router = APIRouter()

# GET /api/usuarios
@router.get("/")
async def listar_usuarios(usuario=Depends(get_user_from_token)):
    await verificaAdmin(usuario)
    return await usuarios_controller.listar()

# GET /api/usuarios/:id
@router.get("/{id}")
async def buscar_usuario(id: str, usuario=Depends(get_user_from_token)):
    await verificaAdmin(usuario)
    return await usuarios_controller.buscar_por_id(id)

# PUT /api/usuarios/:id
@router.put("/{id}")
async def atualizar_usuario(id: str, data: UserUpdate, usuario=Depends(get_user_from_token)):
    await verificaAdmin(usuario)
    return await usuarios_controller.atualizar(id, data)

# DELETE /api/usuarios/:id
@router.delete("/{id}")
async def deletar_usuario(id: str, usuario=Depends(get_user_from_token)):
    await verificaAdmin(usuario)
    return await usuarios_controller.deletar(id)
