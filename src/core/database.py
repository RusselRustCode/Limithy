from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from config.settings import settings


class DataBase:

    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None


def connect_to_mongo():
    print(f"{settings.MONGODB_URL}")
    DataBase.client = AsyncIOMotorClient(settings.MONGODB_URL)
    DataBase.db = AsyncIOMotorDatabase(settings.MONGODB_NAME)
    print(f"Подключено к {settings.MONGODB_NAME}")

def close_mongo_db():
    if DataBase.client:
        DataBase.client.close()
        print("Соединение закрыто")

def get_database() -> AsyncIOMotorDatabase:
    if DataBase.db is None:
        raise ConnectionError("БД не инициализированна")
    return DataBase.db
    
