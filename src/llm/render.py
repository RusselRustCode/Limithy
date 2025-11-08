import json
import os
from typing import Dict, Any
from string import Template
from src.core.models import ContentParams, TestParams, OutputFormat


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
        "blum_level": params.cognitive_level.value,
        "distractor_focus": params.distractor_error_type.value,
        "NumberOfChoises": f"{params.number_of_choices} Варианта(один правильный и {params.number_of_choices - 1} дистрактора)",
        "context_requirement": params.context_requirement.value,
        "difficulity_level": params.difficulity_level.value
    }

    return mapping

def render_prompt_template(template_dict: Dict[str, Any], **kwargs) -> str:
    template_str = json.dumps(template_dict, ensure_ascii=False, indent=2)
    template = Template(template_str)
    filled_str = template.safe_substitute(**kwargs)
    return filled_str