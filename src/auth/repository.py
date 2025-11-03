from typing import Optional
from src.core.database import get_database
from src.core.models import TeacherDB, TeacherCreate
from motor.motor_asyncio import AsyncIOMotorDatabase
from passlib.context import CryptContext
from datetime import datetime
class AuthRepository:

    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
        self.collection = self.db.users
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


    def hash_password(self, password: str) -> str:
        # Обрезаем до 72 байт в кодировке UTF-8
        if len(password.encode('utf-8')) > 72:
            # Обрезаем байты, а не символы
            password_bytes = password.encode('utf-8')[:72]
            password = password_bytes.decode('utf-8', errors='ignore')
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return self.pwd_context.verify(plain_password, hashed_password)

    async def get_teacher_by_email(self, email: str) -> Optional[TeacherDB]:
        user_doc = await self.collection.find_one({"email": email})
        if user_doc:
            return TeacherDB(**user_doc)
        return None
    

    async def create_teacher(self, teacher: TeacherCreate) -> TeacherDB:

        hashed_password = self.hash_password(teacher.password)

        teacher_doc = {
            "email": teacher.email,
            "full_name": teacher.full_name,
            "hashed_password": hashed_password,
            "created_time": datetime.utcnow()
        }

        result = await self.collection.insert_one(teacher_doc)

        new_teacher = await self.collection.find_one({"_id": result.inserted_id})
        return TeacherDB(**new_teacher)