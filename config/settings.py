from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Класс для загрузки конфигурации из переменных окружения.
    Pydantic автоматически ищет переменные с соответствующими именами.
    """
    
    # Конфигурация MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_NAME: str = "llm_analysis_db"

    # Настройки для FastAPI
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "LLM Analysis Gateway"
    
    # Модель для указания файла .env
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

settings = Settings()