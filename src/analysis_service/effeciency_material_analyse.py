import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta


class MaterialEffectivenessAnalyzer:
    def __init__(self, student_logs_df: pd.DataFrame, materials_metadata: Dict[str, Any] = None):
        self.df = student_logs_df
        self.materials_meta = materials_metadata or {}
        self.course_avg_success = self.df['is_correct'].mean() if not self.df.empty else 0.0

    def calculate_material_metrics(self) -> Dict[str, Any]:
        if self.df.empty:
            return {"error": "no_data"}

        material_stats = {}

        for material_id in self.df['material_id'].unique():
            material_data = self.df[self.df['material_id'] == material_id]

            if material_data.empty:
                material_stats[material_id] = self.get_empty_stats()
                continue

            # 1. ЭФФЕКТИВНОСТЬ ОБУЧЕНИЯ
            success_rate = material_data['is_correct'].mean()
            avg_attempts = material_data['attempts'].mean()

            # 2. ВОВЛЕЧЕННОСТЬ С МАТЕРИАЛОМ
            avg_time_on_material = material_data['time_spent_on_m'].mean()
            utilization_ratio = self.calculate_utilization_ratio(material_data)

            # 3. СЛОЖНОСТЬ
            difficulty_index = 1 - success_rate
            time_intensity = material_data['time_spent_on_q'].mean()

            # 4. ДИСТРАКТОРНЫЙ АНАЛИЗ
            distractor_result = self.analyze_distractors(material_data)
            top_distractors = distractor_result.get("top_distractors", [])

            # 5. ПРОГРЕССИЯ
            learning_curve = self.calculate_learning_curve(material_data)

            # 6. СРАВНЕНИЕ С КУРСОМ (опционально)
            success_vs_course = success_rate - self.course_avg_success

            stats = {
                'коэфф успешности': float(success_rate),
                'ср кол-во попыток': float(avg_attempts),
                'ср время на материал': float(avg_time_on_material),
                'коэфф использования матреиала(больше 5 минут) ': float(utilization_ratio),
                'индекс сложности': float(difficulty_index),
                'ср время ответа на вопрос в секундах': float(time_intensity),
                'кривая обучения': float(learning_curve),
                'коэфф успешности vs ср по курсу': float(success_vs_course),

                # Дистракторы
                'top_distractors': top_distractors,
                'частота выбора дистрактора': distractor_result.get("distractor_frequency", {}),
                'неудачные попытки': distractor_result.get("total_wrong_attempts", 0),

                # Метаданные
                'общее взаимодействие': len(material_data),
                'уникальные студенты': int(material_data['student_id'].nunique())
            }

            material_stats[material_id] = stats

        return material_stats

    def get_empty_stats(self) -> Dict[str, Any]:
      return {
            'коэфф успешности': None,
            'ср кол-во попыток': None,
            'ср время на материал': None,
            'коэфф использования матреиала(больше 5 минут) ': None,
            'индекс сложности': None,
            'ср время ответа на вопрос в секундах': None,
            'кривая обучения': None,
            'коэфф успешности vs ср по курсу': None,
            'top_distractors': [],
            'частота выбора дистрактора': {},
            'неудачные попытки': 0,
            'общее взаимодействие': 0,
            'уникальные студенты': 0
        }



    def calculate_utilization_ratio(self, material_data: pd.DataFrame) -> float:
      if material_data.empty:
            return 0.0
      utilized = material_data[material_data['time_spent_on_m'] > 300]
      return len(utilized) / len(material_data)

    def analyze_distractors(self, material_data: pd.DataFrame) -> Dict[str, Any]:
      wrong = material_data[~material_data['is_correct']]

      if wrong.empty:
          return {
              "top_distractors": [],
              "distractor_frequency": {},
              "total_wrong_attempts": 0
          }

      counts = wrong['selected_distractor'].value_counts()
      # Убираем NaN и None
      counts = counts.dropna()
      top_5 = counts.head(5)

      return {
          "top_distractors": top_5.index.tolist(),
          "distractor_frequency": top_5.to_dict(),
          "total_wrong_attempts": int(len(wrong))
      }

    def calculate_learning_curve(self, material_data: pd.DataFrame) -> float:
      if len(material_data) < 2:
          return 0.0

      # Сортируем по времени и группируем по студенту
      df_sorted = material_data.sort_values(['student_id', 'timestamped'])
      df_sorted['attempt_order'] = df_sorted.groupby('student_id').cumcount() + 1

      # Ограничиваем первыми 5 попытками для стабильности
      df_limited = df_sorted[df_sorted['attempt_order'] <= 5]

      if len(df_limited) < 2:
          return 0.0

      # Агрегируем correctness по порядку попытки
      trend = df_limited.groupby('attempt_order')['is_correct'].mean()
      if len(trend) < 2:
          return 0.0

      x = trend.index.values
      y = trend.values
      try:
          slope = np.polyfit(x, y, 1)[0]
          return float(np.clip(slope * 10, -1.0, 1.0))
      except (np.linalg.LinAlgError, ValueError):
          return 0.0
