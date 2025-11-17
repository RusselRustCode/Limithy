from fastapi import APIRouter, HTTPException
from src.llm.external_api import generate_llm_content
from src.core.models import ContentParams, TestParams


llm_router = APIRouter(prefix='/llm', tags=['LLM'])



@llm_router.post('/generate_llm_test')
async def generate_llm_test(test_params: TestParams):
    try:
        result = await generate_llm_content(test_params)
        return {"generate_test": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации тест: {e}")

@llm_router.post('/generate_content')
async def generate_llm_explanation(explanation_params: ContentParams):
    try:
        result = await generate_llm_content(explanation_params)
        return {"generate_explanation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации объяснения: {e}")
 
@llm_router.post('/generate_example')
async def generate_llm_example():
    ...
    
@llm_router.post('/generate_topics')
async def generate_llm_topics():
    ...

@llm_router.post('/generate_terms')    
async def generate_llm_terms():
    ...