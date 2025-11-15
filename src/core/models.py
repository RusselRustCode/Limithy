from pydantic import BaseModel, Field, EmailStr, BeforeValidator
from bson import ObjectId
from enum import Enum
from datetime import datetime
from typing import List, Optional, Annotated

def convert_objectid(v):
    if isinstance(v, ObjectId):
        return str(v)
    return v

PyObjectId = Annotated[str, BeforeValidator(convert_objectid)] #Чекнуть BeforeValidator
# ----------------- МОДЕЛИ АУТЕНТИФИКАЦИИ -----------------
class TeacherLogin(BaseModel):
    email: EmailStr = Field(...)
    password: str = Field(...)

class TeacherCreate(BaseModel):
    email: EmailStr = Field(...)
    full_name: str = Field(...)
    password: str = Field(..., min_length=1)

class TeacherDB(BaseModel):
    id: PyObjectId = Field(alias="_id")
    email: EmailStr
    full_name: str
    hashed_password: str
    created_time: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True


class LLMGeneratedContent(BaseModel):
    reasoning: str
    content: dict

#Параметры для промпта, который передастяс LLM
class OutputFormat(str, Enum):
    MARKDOWN = "Markdown"
    LATEX = "LaTeX"

class TargetAudience(str, Enum):
    MATHEMATICIANS = "для математиков (строгий, с доказательствами)"
    NON_MATHEMATICIANS = "для нематематиков (с практическим применением)"

class UsageToggle(str, Enum):
    YES = "Да"
    NO = "Нет"

class LanguageStyle(str, Enum):
    ACADEMIC = "Академический"
    CASUAL = "Разговорный"
    FORMAL = "Официальный"
    STRICT = "Строгий"

class ExplanationLenght(str, Enum):
    BRIEF = "Кратко (около 100 слов)"
    MEDIUM = "Средне (около 250 слов)"
    DETAILED = "Подробно (около 500 слов)"

class ExampleType(str, Enum):
    THEORETICAL = "Теоретический (классические примеры)"
    REAL_CASE = "Реальный кейс (физика/экономика/астрономия)"



# ------------------
class ContentParams(BaseModel):

    output_format: OutputFormat = Field(
        ...,
        description="Формат, в котором LLM должен представить итоговый контент (Markdown или LaTeX)."
    )

    target_audience: TargetAudience = Field(
        ...,
        description="Уровень сложности и акцент на разных сферах"
    )

    usage_toggle: UsageToggle = Field(
        ...,
        description="Использовать ли метафоры для облегчения усвоения материала"
    )

    historical_content: UsageToggle = Field(
        ...,
        description="Использовать ли исторический контекст"
    )

    language_style: LanguageStyle = Field(
        ...,
        description="Стиль и обзий тон объяснения "
    )

    explanation_len: ExplanationLenght = Field(
        ...,
        description="В каком формате(коротко, средне, детально)"
    )

    exampel_type: ExampleType = Field(
        ...,
        description="Фокусировка примеров (на теории или на реальных жизненных/научных кейсах)"
    )
#-------------------------
#Добавить компоненты для Промта по генерации тестов(вопросов)
class QuestionFormat(str, Enum):
    SINGLE_CHOICE = "Один правильный ответ (Multiple Choice Single Answer)"
    MULTIPLE_CHOICE = "Несколько правильных ответов (Multiple Choice Multiple Answer)"
    OPEN_ENDED = "Открытый вопрос (Числовой ответ или короткий текст)"

class CognitiveLevel(str, Enum):
    RECALL = "Запоминание/Воспроизведение (простые определения)"
    APPLICATION = "Применение/Вычисления (использование формул, базовые задачи)"
    ANALYSIS = "Анализ/Синтез (сложные, многошаговые задачи)"

class DistractorErrorType(str, Enum):
    CONCEPTUAL = "Концептуальные (ошибки в понимании принципов и определений)"
    CALCULATION = "Вычислительные (ошибки в арифметике или подстановке)"
    MISCONCEPTION = "Распространенные заблуждения (типичные ошибки начинающих)"

class NumberOfChoices(int, Enum):
    LOW = 3  # (2 дистрактора)
    MEDIUM = 4 # (3 дистрактора)
    HIGH = 5   # (4 дистрактора)


class ContextRequirement(str, Enum):
    ABSTRACT = "Абстрактный (чистая математическая формулировка)"
    SCENARIO = "Сценарный (вопрос должен быть обернут в реальную/прикладную историю)"

class DifficulityLevel(str, Enum):
    EASY = "Легкий"
    MEDIUM = "Средний"
    HARD = "Сложный"
#-------------------------

class TestParams(BaseModel):
    question_format: QuestionFormat
    cognitive_level: CognitiveLevel
    distractor_error_type: DistractorErrorType
    number_of_choices: NumberOfChoices
    context_requirement: ContextRequirement
    difficulity_level: DifficulityLevel
    
class TopicsParams(BaseModel):
    number_of_topics: int = Field(..., description="Количество тем")
    
class TermsParams(BaseModel):
    topic_title: str = Field(..., description="Название темы")
    numbers_of_terms: int = Field(..., description="Кол-во терминов")
    
class ExampleParams(BaseModel):
    term_name: str = Field(..., description="Название темы")
    explanation_body: str = Field(..., description="Текст объяснения")
    subject_specialization: str = Field(..., description="Специализация")
    

#-------------------------
#Добавить компоненты для TraceLog
#-------------------------
class TraceLog(BaseModel):
    student_id: str = Field(..., description="id Студента")
    material_id: str = Field(..., description="id учебного материала")
    question_id: str = Field(..., description="id Вопроса")

    attempts: int = Field(..., description="Количество попыток")
    is_correct: bool = Field(..., description="Правильность ответа")
    time_spent_on_q: int = Field(..., description="Время потраченное на вопрос(секунды)")
    time_spent_on_m: int = Field(..., description="Время потраченное на материал(в секундах)")

    selected_distractor: Optional[str] = Field(None, description="Текст выбранного дистрактора")

    viewed_material_before: bool = Field(..., description="Смотрел ли материал до этого")
    timestamped: datetime = Field(default_factory=datetime.utcnow, description="Время записи лога")

#-------------------------
#Добавить компоненты для AnalyseResult
class StudentCluster(BaseModel):
    type: str = Field(..., description="")
    cluster: int = Field(..., description="К какому кластеру принадлежит студент")

class EffectivenessSummary(BaseModel):
    best_params: "ContentParams" = Field(..., description="Рекомендованные LLM-параметры для улучшения результатов в этой теме.")
    impact_score: float = Field(..., description="Насколько эти параметры повышают средний балл (в % или скоринге).")

class EngagementAnalyse(BaseModel):
    peak_time_hour: int = Field(..., description="")
    time_vs_success_correlation: float = Field(..., )
#-------------------------
class AnalyseResult(BaseModel):
    last_analysis_date: datetime = Field(default_factory=datetime.utcnow, description="")
    student_id: str = Field(..., description="ID студента")
    topic_id: str = Field(..., description="ID темы")


    studentcluster: StudentCluster = Field(..., description="")
    effectiveness: EffectivenessSummary = Field(..., description="")
    engagement: EngagementAnalyse = Field(..., description="")

    top_distractors_list: List[str] = Field(..., description="Список 5 самых проблемных дистракторов в этой теме/группе.")

AnalyseResult.model_rebuild()

