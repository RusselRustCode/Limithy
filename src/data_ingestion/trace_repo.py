from src.core.database import get_database
from src.core.models import TraceLog
from motor.motor_asyncio import AsyncIOMotorDatabase
from motor.motor_asyncio import AsyncIOMotorChangeStream
from bson import ObjectId
from typing import List
import asyncio
import time
class TraceRepository:

    """
    Класс для работы с коллекцие TraceLog
    """

    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
        self.collection = self.db.student_trace_log

    async def get_new_log(self, change: dict):
        if change.get('operationType') == 'insert':
            log_data_dict = change.get('fullDocument')


            try: 
                new_log = TraceLog(**log_data_dict)
                print(f"Получен новый лог {new_log.timestamped} от студента с id = {new_log.student_id}. Работа Analyse_Service")
                #TODO: #Разработать логику анализ сервиса
            except Exception as e:
                print(f"Ошиба при обработке документа (ChangeStream): {e}")

    async def watch_log(self):
        try:
            pipeline = [{"$match": {"operationType": {"$in": ["insert"]}}}]
            async with self.collection.watch(pipeline=pipeline, full_document='required') as stream:
                print("Change Stream запущен: Ожидание новых логов")

                async for change in stream:
                    asyncio.create_task(self.get_new_log(change))

        except Exception as e:
            print(f"Ошибка {e}! Перезапуск через 3 секунды")
            await asyncio.sleep(3)
            asyncio.create_task(self.watch_log())
