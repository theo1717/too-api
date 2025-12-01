from fastapi import HTTPException

async def verificaAdmin(usuario):
    if not usuario.get("is_admin", False):
        raise HTTPException(status_code=403, detail="Apenas administradores podem acessar.")
    return True
