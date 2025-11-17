from fastapi import APIRouter
from src.analysis_service.analyse_repo import AnalyseRepository

analyse_router = APIRouter()

def get_analyse_repo() -> AnalyseRepository:
    return AnalyseRepository()



@analyse_router.get("analyse/cluster")
async def get_cluster_result():
    ...
    
@analyse_router.get("analyse/engagement_analyse")
async def get_engagement_result():
    ...
    
@analyse_router.get("analyse/material_effectiveness")
async def get_material_result():
    ...