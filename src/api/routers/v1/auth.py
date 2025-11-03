from fastapi import APIRouter, Depends, HTTPException, status
from authx import AuthX, AuthXConfig

from src.core.models import TeacherLogin, TeacherCreate, TeacherDB
from src.auth.repository import AuthRepository
from config.settings import settings
router = APIRouter()

def get_auth_repo() -> AuthRepository:
    return AuthRepository()


@router.post("/auth/register", response_model=TeacherDB, status_code=status.HTTP_201_CREATED)
async def register_teach(
    teacher: TeacherCreate,
    repo: AuthRepository =  Depends(get_auth_repo)
):
    existing_teacher = await repo.get_teacher_by_email(teacher.email)
    if existing_teacher:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь с таким email существует"
        )
    new_teacher = await repo.create_teacher(teacher)
    return new_teacher


@router.post("/auth/login")
async def login_teacher(
    login_data: TeacherLogin,
    repo: AuthRepository = Depends(get_auth_repo)
):
    teacher = await repo.get_teacher_by_email(email=login_data.email)
    if not teacher:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Преподаватель не найден"
        )

    if not repo.verify_password(login_data.password, teacher.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный пароль"
        )
    
    access_token = AuthX(AuthXConfig(JWT_SECRET_KEY = settings.JWT_SECRET_KEY,
    JWT_ALGORITHM = settings.JWT_ALGORITHM,
    JWT_TOKEN_LOCATION = ["cookies"],)).create_access_token(uid=str(teacher.id))
    return {"access_token": access_token, "token_type": "bearer"}