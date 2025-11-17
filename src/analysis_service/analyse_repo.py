from typing import List, Optional
from bson import ObjectId
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase
from src.core.database import get_database
from src.core.models import AnalyseResult, StudentCluster, EffectivenessSummary, EngagementAnalyse


class AnalyseRepository:

    def __init__(self):
        self.db: AsyncIOMotorDatabase = get_database()
        self.collection = self.db.analyse_summary

    #Пока что реализуем самые простые и необходимые методы
    async def save_analysis_result(self, result: AnalyseResult) -> str:

        doc = result.model_dump(by_alias=True, exclude_unset=True)
        result = await self.collection.insert_one(doc)
        return str(result.inserted_id)
    
    async def get_latest_analysis_for_student(self, student_id: str) -> Optional[AnalyseResult]:
        query = {"student_id": student_id}
        doc = await self.collection.find(query).sort("last_analyse_date", -1).limit(1).to_list(length=4)
        if doc:
            return AnalyseResult(**doc[0])
        return None
    
    async def get_latest_analysis_for_topic(self, topic_id: str) -> Optional[AnalyseResult]:
        query = {"topic_id": topic_id}
        doc = await self.collection.find(query).sort("last_analyse_date", -1).limit(1).to_list(length=4)
        if doc:
            return AnalyseResult(**doc[0])
        return None

    async def get_all_analysis_result(self) -> List[AnalyseResult]:
        cursor = self.collection.find().sort("last_analysis_date", -1)
        docs = await cursor.to_list(length=None)
        return [AnalyseResult(**doc) for doc in docs]
    
