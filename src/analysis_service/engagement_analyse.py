import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class EngagementAnalyzer():

  def __init__(self, student_logs_df: pd.DataFrame):
    self.df = student_logs_df
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
      student_data = self.df[self.df['student_id'] == student_id]

      student_data['hour'] = student_data['timestamp'].dt.hour
      student_data['day_of_week'] = student_data['timestamp'].dt.dayofweek
      student_data['is_weekend'] = student_data['day_of_week'].isin([5, 6])

      hour_counts = student_data['hour'].value_counts()
      preferred_hour = hour_counts.index[0] if len(hour_counts) > 0 else 12

      day_distribution = student_data['day_of_week'].value_counts(normalize=True) #Вычисляет долю каждого дня когда студент учился
      regularity_score = day_distribution.std() if len(day_distribution) > 1 else 0.5 #вычисляет std, если регулярно занимался(равномерно), то коэфф будет низкий 


      temp_patterns['временные паттерны'] = {
          'Самый частые часы': preferred_hour,
          'коэфф активность на выходных': student_data['is_weekend'].mean().round(3),
          'уровени регулярности': 1 - round(regularity_score, 3),
          'самый активный день': student_data['day_of_week'].mode()[0] if len(student_data) > 0 else 0,
          'средняя длина сессии': self._calculate_avg_session_length(student_data),
          'Дисперсия активности': student_data['hour'].std() if len(student_data) > 1 else 0
      }

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
    eff = self.engagement_metrics['efficiency'][['коэфф прогресса', 'эффективность обучения', 'performance_stability']]
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

  def calculate_stability_score(self, student_data: pd.DataFrame) -> float:
    if len(student_data) < 2:
        return 0.5
    std = student_data['correctness'].std()
    return max(0.0, 1 - std)

  def _calculate_retention_rate(self, student_data: pd.DataFrame) -> float:
    return student_data['correctness'].mean() if len(student_data) > 0 else 0.5

  def get_student_engagement_summary(self, student_id: str):
    summary = {}
    for metric_type, df in self.engagement_metrics.items():
        if metric_type not in ['activity_dates'] and student_id in df.index:
            summary[metric_type] = df.loc[student_id].to_dict()
    return summary if summary else None



class EngagementVisualizer:
    def __init__(self, engagement_analyzer):
        self.analyzer = engagement_analyzer
        self.metrics = engagement_analyzer.engagement_metrics
    
    def create_comprehensive_dashboard(self):
        """Создание комплексной дашборда вовлеченности"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. ОБЗОРНАЯ СТАТИСТИКА
        self._plot_overview_stats(fig, 231)
        
        # 2. РАСПРЕДЕЛЕНИЕ АКТИВНОСТИ
        self._plot_activity_distribution(fig, 232)
        
        # 3. ВРЕМЕННЫЕ ПАТТЕРНЫ
        self._plot_temporal_patterns(fig, 233)
        
        # 4. ЭФФЕКТИВНОСТЬ ОБУЧЕНИЯ
        self._plot_learning_efficiency(fig, 234)
        
        # 5. СТУДЕНТЫ ГРУППЫ РИСКА
        self._plot_risk_analysis(fig, 235)
        
        # 6. КОРРЕЛЯЦИИ МЕТРИК
        self._plot_correlation_heatmap(fig, 236)
        
        plt.tight_layout()
        return fig
    
    def _plot_overview_stats(self, fig, position):
        ax = fig.add_subplot(position)
        
        total_students = len(self.metrics['activity'])
        active_students = len(self.metrics['activity'][
            self.metrics['activity']['events_per_day'] > 0.5
        ])
        risk_students = len(self.analyzer.risk_students)
        
        stats_data = [total_students, active_students, risk_students]
        stats_labels = ['Всего студентов', 'Активные', 'Группа риска']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = ax.bar(stats_labels, stats_data, color=colors, alpha=0.7)
        ax.set_title('Обзорная статистика вовлеченности')
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, stats_data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                   f'{value}', ha='center', va='bottom')
    
    def _plot_activity_distribution(self, fig, position):
        ax = fig.add_subplot(position)
        
        events_per_day = self.metrics['activity']['events_per_day'].dropna()
        
        ax.hist(events_per_day, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(events_per_day.mean(), color='red', linestyle='--', 
                  label=f'Среднее: {events_per_day.mean():.2f}')
        
        ax.set_xlabel('Событий в день')
        ax.set_ylabel('Количество студентов')
        ax.set_title('Распределение ежедневной активности')
        ax.legend()
    
    def _plot_temporal_patterns(self, fig, position):
        ax = fig.add_subplot(position)
        
        # Активность по часам
        hour_distribution = self.metrics['temporal_patterns']['preferred_study_hour'].value_counts().sort_index()
        
        ax.bar(hour_distribution.index, hour_distribution.values, 
               color='orange', alpha=0.7)
        ax.set_xlabel('Час дня')
        ax.set_ylabel('Количество студентов')
        ax.set_title('Предпочтительное время обучения')
        ax.set_xticks(range(0, 24, 2))
    
    def _plot_learning_efficiency(self, fig, position):
        ax = fig.add_subplot(position)
        
        efficiency = self.metrics['efficiency']['learning_efficiency'].dropna()
        correctness = self.metrics['activity']['avg_correctness'].dropna()
        
        scatter = ax.scatter(efficiency, correctness, alpha=0.6, 
                           c=self.metrics['activity']['total_learning_time'], 
                           cmap='viridis', s=50)
        
        ax.set_xlabel('Эффективность обучения')
        ax.set_ylabel('Средняя правильность')
        ax.set_title('Эффективность vs Правильность')
        plt.colorbar(scatter, ax=ax, label='Общее время обучения')
    
    def _plot_risk_analysis(self, fig, position):
        ax = fig.add_subplot(position)
        
        if not self.analyzer.risk_students.empty:
            risk_factors = self.analyzer.risk_students.select_dtypes(include=[np.number]).mean()
            top_risk_factors = risk_factors.nlargest(5)
            
            ax.barh(range(len(top_risk_factors)), top_risk_factors.values, 
                   color='red', alpha=0.6)
            ax.set_yticks(range(len(top_risk_factors)))
            ax.set_yticklabels([self._format_metric_name(name) for name in top_risk_factors.index])
            ax.set_title('Топ факторы риска')
        else:
            ax.text(0.5, 0.5, 'Нет студентов группы риска', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Анализ рисков')
    
    def _plot_correlation_heatmap(self, fig, position):
        ax = fig.add_subplot(position)
        
        combined_metrics = self.analyzer._combine_all_metrics()
        numeric_columns = combined_metrics.select_dtypes(include=[np.number]).columns
        
        # Берем только основные метрики для читаемости
        selected_metrics = [col for col in numeric_columns if any(x in col for x in 
                            ['events_per_day', 'correctness', 'efficiency', 'engagement', 'risk_score'])]
        
        if len(selected_metrics) > 1:
            correlation_matrix = combined_metrics[selected_metrics].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=ax, fmt='.2f')
            ax.set_title('Корреляции метрик вовлеченности')
        else:
            ax.text(0.5, 0.5, 'Недостаточно данных\nдля корреляционного анализа', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _format_metric_name(self, name):
        """Форматирование названий метрик для читаемости"""
        name_parts = name.split('_')
        return ' '.join(name_parts[-2:]).title()


def create_test_student_logs(num_students=50, num_days=30):
    """Создание тестового датасета логов студентов"""
    
    # Базовые настройки
    start_date = datetime(2024, 1, 1)
    student_ids = [f"student_{i:03d}" for i in range(1, num_students + 1)]
    material_ids = [f"material_{i:03d}" for i in range(1, 21)]
    question_ids = [f"question_{i:03d}" for i in range(1, 101)]
    distractors = ['A', 'B', 'C', 'D', 'conceptual_error', 'calculation_error', 'misunderstanding']
    
    logs = []
    
    for student_id in student_ids:
        # Определяем тип студента (для создания разных паттернов)
        student_type = np.random.choice(['active_high', 'active_medium', 'active_low', 'irregular', 'dropout'], 
                                      p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        # Генерируем логи в зависимости от типа студента
        if student_type == 'active_high':
            num_sessions = np.random.randint(20, 30)
            correctness_range = (0.7, 0.95)
            time_on_material_range = (300, 1200)  # 5-20 минут
        elif student_type == 'active_medium':
            num_sessions = np.random.randint(15, 25)
            correctness_range = (0.5, 0.8)
            time_on_material_range = (180, 600)   # 3-10 минут
        elif student_type == 'active_low':
            num_sessions = np.random.randint(8, 15)
            correctness_range = (0.3, 0.6)
            time_on_material_range = (60, 300)    # 1-5 минут
        elif student_type == 'irregular':
            num_sessions = np.random.randint(5, 12)
            correctness_range = (0.2, 0.7)
            time_on_material_range = (30, 400)    # 0.5-6 минут
        else:  # dropout
            num_sessions = np.random.randint(1, 5)
            correctness_range = (0.1, 0.4)
            time_on_material_range = (10, 120)    # мало времени
            
        # Создаем сессии для студента
        for session in range(num_sessions):
            # Случайная дата в пределах периода
            days_offset = np.random.randint(0, num_days)
            session_date = start_date + timedelta(days=days_offset)
            
            # Время суток (предпочтения по типам)
            if student_type in ['active_high', 'active_medium']:
                hour = np.random.normal(19, 3)  # вечерние
            else:
                hour = np.random.uniform(9, 23)  # случайные
            
            hour = max(8, min(23, int(hour)))
            minute = np.random.randint(0, 60)
            
            session_time = session_date.replace(hour=hour, minute=minute)
            
            # 1-5 действий в сессии
            num_actions = np.random.randint(1, 6)
            
            for action in range(num_actions):
                material_id = np.random.choice(material_ids)
                question_id = np.random.choice(question_ids)
                
                # Время на материал (секунды)
                time_on_material = np.random.randint(time_on_material_range[0], time_on_material_range[1])
                
                # Правильность ответа
                base_correctness = np.random.uniform(correctness_range[0], correctness_range[1])
                
                # Влияние времени на материал на правильность
                material_effect = min(0.3, time_on_material / 4000)
                final_correctness = min(0.95, base_correctness + material_effect)
                
                is_correct = np.random.random() < final_correctness
                
                # Количество попыток
                if is_correct:
                    attempts = np.random.randint(1, 3)
                else:
                    attempts = np.random.randint(2, 5)
                
                # Время на вопрос
                base_time = np.random.randint(30, 300)
                # Чем больше попыток, тем больше общее время
                time_on_question = base_time * attempts * np.random.uniform(0.8, 1.2)
                
                # Дистрактор (только для неправильных ответов)
                selected_distractor = None if is_correct else np.random.choice(distractors)
                
                # Создаем запись лога
                log_entry = {
                    'student_id': student_id,
                    'material_id': material_id,
                    'question_id': question_id,
                    'timestamp': session_time + timedelta(minutes=action*5),
                    'time_spent_on_mat': max(10, int(time_on_material * np.random.uniform(0.7, 1.3))),
                    'correctness': is_correct,
                    'attempts': attempts,
                    'time_spent_on_question': max(15, int(time_on_question)),
                    'selected_distractor': selected_distractor,
                    'student_type': student_type  # для проверки качества кластеризации
                }
                
                logs.append(log_entry)
    
    return pd.DataFrame(logs)

# # Создаем тестовый датасет
# print("🎲 Создание тестового датасета student_logs_df...")
# student_logs_df = create_test_student_logs(num_students=50, num_days=30)

# print("✅ Датасет создан!")
# print(f"📊 Размер датасета: {student_logs_df.shape}")
# print(f"👥 Уникальных студентов: {student_logs_df['student_id'].nunique()}")
# print(f"📅 Период данных: {student_logs_df['timestamp'].min()} - {student_logs_df['timestamp'].max()}")
# print(f"📝 Всего записей: {len(student_logs_df)}")

# # Покажем первые несколько строк
# print("\nПервые 5 записей:")
# print(student_logs_df.head().to_string())

# # Базовая статистика
# print("\n📈 Базовая статистика:")
# print(f"Среднее время на материал: {student_logs_df['time_spent_on_mat'].mean():.1f} сек")
# print(f"Среднее время на вопрос: {student_logs_df['time_spent_on_question'].mean():.1f} сек")
# print(f"Общая правильность: {student_logs_df['correctness'].mean():.2%}")
# print(f"Среднее количество попыток: {student_logs_df['attempts'].mean():.2f}")

# # Распределение по типам студентов
# print("\n🎯 Распределение студентов по типам:")
# print(student_logs_df.groupby('student_type')['student_id'].nunique())

# # Инициализация и расчет
# engagement_analyzer = EngagementAnalyzer(student_logs_df)
# engagement_metrics = engagement_analyzer.calculate_comprehensive_metrics()

# # Визуализация
# visualizer = EngagementVisualizer(engagement_analyzer)
# dashboard = visualizer.create_comprehensive_dashboard()
# plt.show()

# # Анализ результатов
# print(f"🎯 Всего студентов: {len(engagement_metrics['activity'])}")
# print(f"⚠️  Студентов в группе риска: {len(engagement_analyzer.risk_students)}")
# print(f"📈 Средняя активность: {engagement_metrics['activity']['events_per_day'].mean():.2f} событий/день")

# # Студенты для особого внимания
# if not engagement_analyzer.risk_students.empty:
#     print("\n🔴 Студенты группы риска:")
#     print(engagement_analyzer.risk_students.index.tolist())