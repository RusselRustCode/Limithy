from typing import List
from src.core.models import TraceLog, AnalyseResult, StudentCluster, EffectivenessSummary
from src.data_ingestion.trace_repo import TraceRepository
from src.analysis_service.analyse_repo import AnalyseRepository
class AnalysisService:


    def __init__(self, trace_repo: TraceRepository, repo: AnalyseRepository = None):
        self.trace_repo = trace_repo
        self.repo = repo or AnalyseRepository()


    async def fetch_and_prepare_data(self, topic_id: str) -> List[TraceLog]:
        logs = self.trace_repo.log_buffer.copy()
        return logs

    async def run_analysis(self, student_id: str, topic_id: str) -> AnalyseResult:
        ...
        #Основной метод для запуска анализа
