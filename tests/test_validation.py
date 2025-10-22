# Файл: tests/test_validation.py

import pytest
from httpx import AsyncClient

# Загружаем фикстуру client из conftest.py

@pytest.mark.asyncio
async def test_trace_log_validation_failure(client: AsyncClient, valid_log_data):
    """
    Проверяет, что API возвращает 422, если поле 'attempts' имеет неверный тип (str вместо int).
    """
    
    # 1. Изменение валидных данных на невалидные
    invalid_data = valid_log_data.copy()
    invalid_data["attempts"] = "одна попытка" # Ожидается int, передаем str

    # 2. Выполнение запроса
    response = await client.post("/v1/trace/log", json=invalid_data)

    # 3. Проверка результата
    assert response.status_code == 422
    
    # 4. Проверка сообщения об ошибке (чтобы убедиться, что Pydantic сработал)
    response_json = response.json()
    assert response_json["detail"][0]["loc"] == ["body", "attempts"]
    assert "value is not a valid integer" in response_json["detail"][0]["msg"]