from fastapi import FastAPI
from config.settings import settings
from core.database import connect_to_mongo, close_mongo_db, get_database
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

@app.on_event("startup")
async def startup_event():
    print("")
    connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_event():
    print(" ")
    close_mongo_db()


app.include_router(
    trace.router,
    prefix=settings.API_V1_STR,
    tags=["TRACE"]
)
