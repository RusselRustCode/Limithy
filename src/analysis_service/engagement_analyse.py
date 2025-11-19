import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


class EngagementAnalyzer():

  def __init__(self, student_logs_df: pd.DataFrame):
    self.df = student_logs_df.copy()
    self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
    self.engagement_metrics = {}
    self.student_risk = {}

  def calculate_engagement_metrics(self):
    self.calculate_active_metrics()
    self.calculate_learning_patterns()
    self.calculate_temp_patterns()
    self.calculate_learning_efficiency()
    self.calculate_risk_scores()

  def calculate_active_metrics(self):
    student_activity = self.df.groupby('student_id').agg({
        'timestamp': ['min', 'max', 'count'],
        'time_spent_on_mat': 'sum',
        'time_spent_on_question': 'sum',
        'attempts': 'sum',
        'correctness': 'mean'
    }).round(2)

    student_activity.columns = ['first_activity', 'last_activity', 'total_events',
                          'total_material_time', 'total_question_time',
                          'total_attempts', 'avg_correctness']

    student_activity['total_learning_time'] = (
            student_activity['total_material_time'] + student_activity['total_question_time']
        )
    student_activity['activity_duration_days'] = (
        (student_activity['last_activity'] - student_activity['first_activity']).dt.days + 1
    )

    student_activity['events_per_day'] = (
    student_activity['total_events'] / student_activity['activity_duration_days']).round(2)

    activity_numeric = student_activity.drop(['first_activity', 'last_activity'], axis=1) #удаляем временные столбцы, т.к. время не используется в вычислениях

    self.engagement_metrics['activity'] = activity_numeric
    self.engagement_metrics['activity_dates'] = student_activity[['first_activity', 'last_activity']]

  def calculate_learning_patterns(self):
    learning_patterns = {}

    for student_id in self.df['student_id'].unique():
      student_data = self.df[self.df['student_id'] == student_id]

      total_material_time = student_data['time_spent_on_mat'].sum()
      total_question_time = student_data['time_spent_on_question'].sum()

      material_engagement = (total_material_time / total_question_time if total_question_time > 0 else 0) #отношение времени на материал / времени на тест

      retry_rate = (student_data['attempts'] > 1).mean() #ср кол-во попыток на тест

      passive_score = self.calculate_passive_score(student_data, 300, 0.5, 100, 0.1)
      effort_eff = self.calculate_effort_efficiency(student_data)


      learning_patterns[student_id] = {
          'Коэффициент_вовлеченности_материала': round(material_engagement, 3),
          'Частота попыток': round(retry_rate, 3),
          'Ср кол-во попыток на вопрос': student_data['attempts'].mean().round(2),
          'consistency_score': self._calculate_consistency(student_data),
          'отношение времени материала на материал + тест': total_material_time / (total_material_time + total_question_time)
          if (total_material_time + total_question_time) > 0 else 0,
          'Индикатор пассивного потребления': round(passive_score, 3),
          'эффективность усилий': round(effort_eff, 3)
      }

      self.engagement_metrics['learning_patterns'] = pd.DataFrame.from_dict(
          learning_patterns, orient='index'
      )

  def calculate_temp_patterns(self):
    temp_patterns = {}

    for student_id in self.df['student_id'].unique():
        student_data = self.df[self.df['student_id'] == student_id].copy()

        if student_data.empty:
            # Обработка случая, если у студента нет записей (маловероятно, но безопасно)
            temp_patterns[student_id] = {
                'Самый частые часы': np.nan,
                'коэфф активность на выходных': 0.0,
                'уровени регулярности': 0.0,
                'самый активный день': np.nan,
                'средняя длина сессии': 0.0,
                'Дисперсия активности': 0.0
            }
            continue

        student_data['hour'] = student_data['timestamp'].dt.hour
        student_data['day_of_week'] = student_data['timestamp'].dt.dayofweek
        student_data['is_weekend'] = student_data['day_of_week'].isin([5, 6])

        hour_counts = student_data['hour'].value_counts()
        preferred_hour = hour_counts.index[0] if len(hour_counts) > 0 else 12

        day_distribution = student_data['day_of_week'].value_counts(normalize=True)
        regularity_score = day_distribution.std() if len(day_distribution) > 1 else 0.5

        temp_patterns[student_id] = {
            'Самый частые часы': preferred_hour,
            'коэфф активность на выходных': student_data['is_weekend'].mean().round(3),
            'уровени регулярности': 1 - round(regularity_score, 3),
            'самый активный день': student_data['day_of_week'].mode()[0],
            'средняя длина сессии': self._calculate_avg_session_length(student_data),
            'Дисперсия активности': student_data['hour'].std() if len(student_data) > 1 else 0
        }

    # Создаём DataFrame с индексом = student_id
    self.engagement_metrics['temp_patterns'] = pd.DataFrame.from_dict(
        temp_patterns, orient='index'
    )

  def calculate_learning_efficiency(self):
    efficiency_metrics = {}

    for student_id in self.df['student_id'].unique():
      student_data = self.df[self.df['student_id'] == student_id]

      avg_correctness = student_data['correctness'].mean()
      total_learning_time = (
          student_data['time_spent_on_mat'].sum() +
          student_data['time_spent_on_question'].sum()
      )

      efficiency = avg_correctness / (total_learning_time / 3600) if total_learning_time > 0 else 0
      progress_score = self.calculate_learning_progress(student_data)
      retention = self.calculate_retention_rate(student_data)
      stability = self.calculate_stability_score(student_data)

      efficiency_metrics[student_id] = {
          'эффективность обучения': round(efficiency, 4),
          'коэфф прогресса': progress_score,
          'knowledge_retention': retention,
          'эффективность по времени': round(avg_correctness / (total_learning_time / 60), 4)
          if total_learning_time > 0 else 0,
          'регулярность занятий': round(stability, 3)
      }

    self.engagement_metrics['efficiency'] = pd.DataFrame.from_dict(
        efficiency_metrics, orient='index'
    )

  def calculate_risk_scores(self):
    try:
      key_metrics_df = self.extract_key_metrics_for_risk()
      if key_metrics_df.empty or len(key_metrics_df) < 3:
            print("Недостаточно данных для анализа рисков")
            key_metrics_df['risk_flag'] = 1
            key_metrics_df['risk_score'] = 0
            self.engagement_metrics['risk_assessment'] = key_metrics_df
            self.risk_students = pd.DataFrame()
            return
      key_metrics_df = key_metrics_df.fillna(0)
      scaler = StandardScaler()
      scaled = scaler.fit_transform(key_metrics_df)

      contamination = min(0.3, max(0.1, 0.5 / len(key_metrics_df)))
      iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=50)
      risk_flags = iso.fit_predict(scaled)
      risk_scores = iso.decision_function(scaled)

      key_metrics_df['risk_flag'] = risk_flags
      key_metrics_df['risk_score'] = risk_scores

      self.engagement_metrics['risk_assessment'] = key_metrics_df
      self.risk_students = key_metrics_df[key_metrics_df['risk_flag'] == -1]

      print(f"Анализ рисков завершён. Студентов в группе риска: {len(self.risk_students)}")

    except Exception as e:
      print(f"Ошибка в анализе рисков: {e}")
      dummy = pd.DataFrame(index=self.engagement_metrics['activity'].index)
      dummy['risk_flag'] = 1
      dummy['risk_score'] = 0
      self.engagement_metrics['risk_assessment'] = dummy
      self.risk_students = pd.DataFrame()


  def extract_key_metrics_for_risk(self) -> pd.DataFrame:
    act = self.engagement_metrics['activity'][['avg_correctness', 'events_per_day']]
    eff = self.engagement_metrics['efficiency'][['коэфф прогресса', 'эффективность обучения', 'регулярность занятий']]
    pat = self.engagement_metrics['learning_patterns'][
        ['Частота попыток', 'Индикатор пассивного потребления', 'эффективность усилий']
    ]
    return pd.concat([act, eff, pat], axis=1).select_dtypes(include=[np.number])


  def calculate_passive_score(self, student_data: pd.DataFrame, time_threshold: int = 300, score_threshold: float = 0.5, time_difference: int = 100, score_differnce: float = 0.1) -> float:
    avg_time_material = student_data['time_spent_on_mat'].mean()
    avg_correct = student_data['correctness'].mean()

    if avg_time_material > time_threshold and avg_correct < score_threshold:
      return 0.9
    elif avg_time_material > time_threshold - time_difference and avg_correct < score_threshold + score_differnce:
      return 0.6
    else:
      return 0.1

  def calculate_effort_efficiency(self, student_data: pd.DataFrame) -> float:
    total_time_q = student_data['time_spent_on_question'].sum()
    total_attempts = student_data['attempts'].sum()
    if total_attempts == 0:
        return 0.0
    efficiency = total_time_q / total_attempts  # секунд на попытку
    return 1 / (1 + efficiency / 60)

  def _calculate_consistency(self, student_data: pd.DataFrame) -> float:
        if len(student_data) < 2:
            return 0.5
        std = student_data['correctness'].std()
        return max(0, 1 - std) if not np.isnan(std) else 0.5

  def calculate_stability_score(self, student_data: pd.DataFrame) -> float:
    if len(student_data) < 2:
        return 0.5
    std = student_data['correctness'].std()
    return max(0.0, 1 - std)

  def calculate_retention_rate(self, student_data: pd.DataFrame) -> float:
    return student_data['correctness'].mean() if len(student_data) > 0 else 0.5

  def _calculate_avg_session_length(self, student_data: pd.DataFrame) -> float:
        if len(student_data) < 2:
            return 0
        sorted_data = student_data.sort_values('timestamp')
        diffs = sorted_data['timestamp'].diff().dt.total_seconds()
        session_lengths = diffs[diffs < 3600]
        return session_lengths.mean() if len(session_lengths) > 0 else 0

  def calculate_learning_progress(self, student_data: pd.DataFrame) -> float:
      if len(student_data) < 3:
          return 0.5
      sorted_data = student_data.sort_values('timestamp')
      x = np.arange(len(sorted_data))
      y = sorted_data['correctness'].values
      try:
          slope = np.polyfit(x, y, 1)[0]
          return max(0, min(1, slope * 10 + 0.5))
      except:
          return 0.5

  def get_student_engagement_summary(self, student_id: str):
    summary = {}
    for metric_type, df in self.engagement_metrics.items():
        if metric_type not in ['activity_dates'] and student_id in df.index:
            summary[metric_type] = df.loc[student_id].to_dict()
    return summary if summary else None

