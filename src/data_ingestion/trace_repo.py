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

    def __init__(self, batch_size: int = 15, timeout_sec: int = 300):
        self.db: AsyncIOMotorDatabase = get_database()
        self.collection = self.db.student_trace_log
        self.batch_size = batch_size
        self.timeout_sec = timeout_sec

        # Буфер для всех логов (не разделяем по студентам)
        self.log_buffer: List[TraceLog] = []
        self.last_processed_time = time.time()
        self.timeout_task: asyncio.Task | None = None

    async def get_new_log(self, change: dict):
        if change.get('operationType') == 'insert':
            log_data_dict = change.get('fullDocument')
            try:
                new_log = TraceLog(**log_data_dict)
                self.log_buffer.append(new_log)
                print(f"Получен лог от студента {new_log.student_id}. Всего в буфере: {len(self.log_buffer)}")

                # # Запускаем таймер при первом логе в буфере
                # if len(self.log_buffer) == 1:
                #     if self.timeout_task and not self.timeout_task.done():
                #         self.timeout_task.cancel()
                #     self.timeout_task = asyncio.create_task(self.timeout_handler()) #Подумать над таймером

                # Если набралось достаточно — обрабатываем
                if len(self.log_buffer) >= self.batch_size:
                    await self.process_batch()

            except Exception as e:
                print(f"Ошибка при обработке документа (ChangeStream): {e}")

    async def timeout_handler(self):
        """Сбрасывает буфер по таймауту, даже если <15 логов."""
        await asyncio.sleep(self.timeout_sec)
        if self.log_buffer:
            print(f"Таймаут: обработка {len(self.log_buffer)} накопленных логов")
            await self.process_batch()

    async def process_batch(self):
        ...

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
