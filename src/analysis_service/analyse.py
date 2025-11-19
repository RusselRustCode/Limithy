from typing import List, Optional
from src.core.models import TraceLog, AnalyseResult, StudentCluster, EffectivenessSummary, EngagementAnalyse
from src.data_ingestion.trace_repo import TraceRepository
from src.analysis_service.analyse_repo import AnalyseRepository
class AnalysisService:


    def __init__(self, trace_repo: TraceRepository, repo: AnalyseRepository = None):
        self.trace_repo = trace_repo
        self.repo = repo or AnalyseRepository()


    async def fetch_and_prepare_data(self, topic_id: str) -> List[TraceLog]:
        logs = self.trace_repo.log_buffer.copy()
        return logs

    async def run_analysis_engagement(self, student_id: str, topic_id: str) -> EngagementAnalyse:
        ...
    async def run_analysis_cluster(self) -> StudentCluster:
        ...

    async def run_analysis_effeciency_material(self) -> EffectivenessSummary:
        ...

    async def join_all_analyse(self) -> AnalyseResult:
        ...