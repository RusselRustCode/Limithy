from fastapi import FastAPI
from config.settings import settings
from src.core.database import connect_to_mongo, close_mongo_db
from src.data_ingestion.trace_repo import TraceRepository
import asyncio
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)




@app.on_event("startup")
async def startup_event():
    print("Hello world")
    connect_to_mongo()

    app.state.trace_repo = TraceRepository()

    app.state.log_watcher_task = asyncio.create_task(app.state.trace_repo.watch_log())
    print("Фоновая задача запущена")

@app.on_event("shutdown")
async def shutdown_event():
    print("Bye world")
    if hasattr(app.state, 'log_watcher_task') and app.state.log_watcher_task:
        app.state.log_watcher_task.cancel()
        print("Фоновая задача отменнена")

    close_mongo_db()
 
# # from routers.v1.analysis import analysis
# # app.include_router(
# #     analysis,
# #     prefix=settings.API_V1_STR,
# #     tags=["ANALYSIS"]
# # )


