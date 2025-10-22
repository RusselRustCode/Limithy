# Файл: tests/conftest.py (для фиктивных данных и клиента)

import sys
from pathlib import Path

# Добавляем корневую директорию в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from httpx import AsyncClient
from src.api.main import app # Ваш основной экземпляр FastAPI
from src.core.models import TraceLog
from unittest.mock import AsyncMock, patch
from datetime import datetime

# Фиктивный репозиторий для изоляции тестов от реальной БД
@pytest.fixture
def mock_trace_repo():
    """Имитация репозитория для проверки вызовов сохранения."""
    mock = AsyncMock()
    return mock

# Асинхронный тестовый клиент для FastAPI
@pytest.fixture
async def client() -> AsyncClient:
    """Создание тестового клиента FastAPI."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

# Корректные данные для записи лога
@pytest.fixture
def valid_log_data():
    """Возвращает JSON с валидными данными для TraceLog."""
    return {
        "student_id": "std_123",
        "material_id": "mat_abc",
        "attempts": 1,
        "is_correct": True,
        "time_spent_on_question": 45,
        "time_spent_on_material": 120,
        "selected_distractor": None,
        "viewed_material_before": True,
        "timestamp": datetime.utcnow().isoformat()
    }