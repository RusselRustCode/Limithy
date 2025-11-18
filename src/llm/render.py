import json
import os
from typing import Dict, Any
from string import Template
from src.core.models import ContentParams, TestParams, OutputFormat, TopicsParams, TermsParams, ExampleParams
from src.core.models import *
import re

PROMPT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_template(template_name: str) -> Dict[str, Any]:
    path = os.path.join(PROMPT_TEMPLATES_DIR, template_name)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def map_content_params_to_template(params: ContentParams) -> Dict[str, str]:
    mapping = {
        "style": params.language_style.value,
        "audience_level": params.target_audience.value,
        "metaphore_use": params.usage_toggle.value,
        "example_focus": params.exampel_type.value,
        "tone": params.language_style.value,
        "content_lenght": params.explanation_len.value,
        "formula_visibility": "Показывать" if params.output_format == OutputFormat.LATEX else "Скрывать" #Додумать
    }

    return mapping

def map_test_params_to_template(params: TestParams) -> Dict[str, str]:
    mapping = {
        "question_format": params.question_format.value,
        "cognitive_level": params.cognitive_level.value,
        "distractor_error_type": params.distractor_error_type.value,
        "number_of_choices": f"{params.number_of_choices} Варианта(один правильный и {params.number_of_choices - 1} дистрактора)",
        "context_requirement": params.context_requirement.value,
        "difficulty_level": params.difficulty_level.value
    }

    return mapping

def map_topics_params_to_template(params: TopicsParams) -> Dict[str, str]:
    mapping = {
        "number_of_topics": params.number_of_topics
    }
    return mapping

def map_terms_params_to_template(params: TermsParams) -> Dict[str, str]:
    mapping = {
        "topic_title": params.topic_title,
        "number_of_terms": params.numbers_of_terms
    }
    return mapping

def map_example_params_to_template(params: ExampleParams) -> Dict[str, str]:
    mapping = {
        "term_name": params.term_name,
        "explanation_body": params.explanation_body,
        "subject_specialization": params.subject_specialization
    }
    return mapping

def render_prompt_template(template_dict: Dict[str, Any], **kwargs) -> str:
    template_str = json.dumps(template_dict, ensure_ascii=False, indent=2)
    template = Template(template_str)
    filled_str = template.safe_substitute(**kwargs)
    return filled_str

def extract_json_from_text(text: str) -> dict:
    """
    Извлекает JSON из текста, который может содержать комментарии и несколько JSON-блоков
    """
    # Ищем JSON-объекты в тексте с помощью регулярного выражения
    json_pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}|{[^{}]*}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if not matches:
        raise ValueError("Не удалось найти JSON в ответе LLM")
    
    # Берем последний найденный JSON (обычно это финальный ответ)
    json_text = matches[-1]
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Если не парсится, попробуем очистить от возможных лишних символов
        json_text = re.sub(r',\s*}', '}', json_text)  # Убираем лишние запятые
        json_text = re.sub(r',\s*]', ']', json_text)
        return json.loads(json_text)

def main():
    temple = load_template("test_generate.json")
    params = TestParams(
        question_format=QuestionFormat.MULTIPLE_CHOICE,
        cognitive_level=CognitiveLevel.ANALYSIS,
        distractor_error_type=DistractorErrorType.CONCEPTUAL,
        number_of_choices=NumberOfChoices.MEDIUM,
        context_requirement=ContextRequirement.SCENARIO,
        difficulty_level=DifficultyLevel.MEDIUM  
    )
    map = map_test_params_to_template(params=params)
    print(map)
    filled_str = render_prompt_template(temple, **map)
    print(filled_str)
    test_string = "question_format: {question_format}, cognitive_level: {cognitive_level}"
    rendered_test = test_string.format(**map)
    print("Test render:", rendered_test)
    
    
if __name__ == "__main__":
    main()