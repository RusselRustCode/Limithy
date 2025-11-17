class MaterialEffectivenessAnalyzer:
    def __init__(self, student_logs_df, materials_metadata):
        self.df = student_logs_df
        self.materials_meta = materials_metadata
    
    def calculate_material_metrics(self):
        material_stats = {}
        
        for material_id in self.df['material_id'].unique():
            material_data = self.df[self.df['material_id'] == material_id]
            
            stats = {
                # 1. ЭФФЕКТИВНОСТЬ ОБУЧЕНИЯ
                'success_rate': material_data['correctness'].mean(),
                'avg_attempts': material_data['attempts'].mean(),
                'completion_rate': self._calculate_completion_rate(material_data),
                
                # 2. ВОВЛЕЧЕННОСТЬ С МАТЕРИАЛОМ
                'avg_time_on_material': material_data['time_spent_on_mat'].mean(),
                'material_utilization_ratio': self._calculate_utilization_ratio(material_data),
                
                # 3. СЛОЖНОСТЬ
                'difficulty_index': 1 - material_data['correctness'].mean(),
                'time_intensity': material_data['time_spent_on_question'].mean(),
                
                # 4. ДИСТРАКТОРНЫЙ АНАЛИЗ
                'distractor_analysis': self._analyze_distractors(material_data),
                
                # 5. ПРОГРЕССИЯ
                'learning_curve': self._calculate_learning_curve(material_data)
            }
            
            material_stats[material_id] = stats
        
        return material_stats
    
    def _calculate_utilization_ratio(self, material_data):
        """Насколько студенты используют материал перед тестом"""
        utilized = material_data[material_data['time_spent_on_mat'] > 60]  # больше 60 секунд
        return len(utilized) / len(material_data) if len(material_data) > 0 else 0
    
    def _analyze_distractors(self, material_data):
        """Анализ каких неправильных ответов выбирают чаще"""
        wrong_answers = material_data[~material_data['correctness']]
        distractor_counts = wrong_answers['selected_distractor'].value_counts()
        return distractor_counts.to_dict()