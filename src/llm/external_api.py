from langchain_core.messages import HumanMessage
from langchain_gigachat.chat_models import GigaChat
from src.llm.render import load_template, map_content_params_to_template, map_test_params_to_template, render_prompt_template, map_example_params_to_template, map_terms_params_to_template, map_topics_params_to_template
import logging
import json
from src.core.models import LLMGeneratedContent, ContentParams, TestParams, TermsParams, TopicsParams, ExampleParams
from fastapi import HTTPException
from pydantic import ValidationError
from typing import TypeVar, Dict, Callable, Tuple, Type
from src.core.models import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ValidModels = TypeVar('ValidModels', ContentParams, TestParams, TermsParams, TopicsParams, ExampleParams)

TEMPLATE_MAPPING: Dict[Type[ValidModels], Tuple[str, Callable]] = {
    ContentParams: ("explanation.json", map_content_params_to_template),
    TestParams: ("test_generate.json", map_test_params_to_template),
    TopicsParams: ("topics.json", map_topics_params_to_template),
    TermsParams: ("terms.json", map_terms_params_to_template),
    ExampleParams: ("problem_example.json", map_example_params_to_template)
}

try: 
    chat = GigaChat(
        credentials="MDFlNjcwMzctNDIzOS00YWIyLTljNzUtOTZlMjI5NjJlZTM3OjlkOWY3ZGExLTIzNmUtNDViYS1hNTJjLTNjMzRlN2ZkZDg5NQ==",
        verify_ssl_certs=False,
        timeout=30
    )
    logger.info("GigaChat инициализирован")
except Exception as e:
    raise logger.error(f"Error: {e}")

def generate_llm_content(content_params: ValidModels) -> LLMGeneratedContent:
    template_name, map_func = TEMPLATE_MAPPING[type(content_params)]
    
    template = load_template(template_name)
    
    map = map_func(content_params)
    
    filled_map = render_prompt_template(template, **map)
    
    messages = [
        HumanMessage(content=filled_map)
    ]
    
    try:
        response = chat.invoke(messages)
        raw_response_text = response.content
        logger.info(f"Ответ LLM: {raw_response_text}")

        try:
            json_response = json.loads(raw_response_text)
        except json.JSONDecodeError:
            logger.error(f"LLM вернул текст, не похожий на JSON: {raw_response_text}")
            raise ValueError("LLM вернул текст, не похожий на JSON.")

        validated_content = LLMGeneratedContent.model_validate(json_response)

        return validated_content

    except ValidationError as ve:
        logger.error(f"Ошибка валидации ответа LLM: {ve}")
        raise HTTPException(status_code=500, detail=f"Ошибка валидации ответа LLM: {ve}")
    except json.JSONDecodeError as je:
        logger.error(f"Ошибка парсинга JSON из ответа LLM: {je}")
        raise HTTPException(status_code=500, detail=f"LLM вернул некорректный JSON.")
    except Exception as e:
        logger.error(f"Ошибка при вызове LLM: {e}")
        raise




def main():
    params = TestParams(
        question_format=QuestionFormat.MULTIPLE_CHOICE,
        cognitive_level=CognitiveLevel.ANALYSIS,
        distractor_error_type=DistractorErrorType.CONCEPTUAL,
        number_of_choices=NumberOfChoices.MEDIUM,
        context_requirement=ContextRequirement.SCENARIO,
        difficulty_level=DifficultyLevel.MEDIUM  
    )
    
    content = generate_llm_content(params)
    print(content)

    
if __name__ == "__main__":
    main()





