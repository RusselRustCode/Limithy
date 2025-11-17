import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_students = 30

data = {
    'student_id': range(n_students),
    #Сделаем так, 4 кластера, посмотрим как он определит

    # Cluster 0: (5 студентов)
    # Cluster 1: (10 студентов)  
    # Cluster 2: (10 студентов)
    # Cluster 3: (5 студентов)
    
    'attempts': np.concatenate([
        np.random.randint(1, 3, 5),   
        np.random.randint(2, 4, 10),   
        np.random.randint(3, 6, 10),   
        np.random.randint(1, 2, 5)     
    ]),
    
    'correctness': np.concatenate([
        np.random.uniform(0.8, 1.0, 5),                                                 
        np.random.uniform(0.6, 0.8, 10),  
        np.random.uniform(0.3, 0.6, 10),  
        np.random.uniform(0.4, 0.5, 5)    
    ]),
    
    'time_spent_on_question': np.concatenate([
        np.random.normal(30, 5, 5),    
        np.random.normal(60, 10, 10),  
        np.random.normal(120, 20, 10), 
        np.random.normal(25, 3, 5)     
    ]),
    
    'time_spent_on_material': np.concatenate([
        np.random.normal(5, 2, 5),     
        np.random.normal(15, 5, 10),   
        np.random.normal(25, 8, 10),   
        np.random.normal(2, 1, 5)      
    ]),
    
    'selected_distractor_freq': np.concatenate([
        np.random.uniform(0.0, 0.1, 5),   
        np.random.uniform(0.1, 0.3, 10),  
        np.random.uniform(0.3, 0.6, 10),  
        np.random.uniform(0.5, 0.7, 5)    
    ]),
    
    'study_time_preference': np.concatenate([
        np.random.normal(14, 2, 5),    
        np.random.normal(19, 3, 10),     
        np.random.normal(10, 2, 10),   
        np.random.normal(23, 1, 5)     
    ])
}

df = pd.DataFrame(data)

# Добавляем щепоточку гауссова шума для реалистичности
for col in ['attempts', 'correctness', 'time_spent_on_question', 
           'time_spent_on_material', 'selected_distractor_freq', 'study_time_preference']:
    df[col] = df[col] + np.random.normal(0, 0.1, n_students)
    

df['correctness'] = np.clip(df['correctness'], 0, 1)
df['selected_distractor_freq'] = np.clip(df['selected_distractor_freq'], 0, 1)
df['attempts'] = np.maximum(1, df['attempts'].round())
df['study_time_preference'] = np.clip(df['study_time_preference'], 0, 24)

print("Первые 5 строк датасета:")
print(df.head().round(2))
print(f"\nРазмер датасета: {df.shape}")



features = df[['attempts', 'correctness', 'time_spent_on_question', 
              'time_spent_on_material', 'selected_distractor_freq', 
              'study_time_preference']]

# Стандартизация - ПИЗДЕЦ ВАЖНО для DBSCAN!
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


dbscan = DBSCAN(eps=1.5, min_samples=3)  # Начнем с этих параметров, пока что лучшие параметры min_samples = 3, а eps = 1.5..1.6
clusters = dbscan.fit_predict(features_scaled)


df['cluster'] = clusters

print("\n=== РЕЗУЛЬТАТЫ DBSCAN ===")
print(f"Найдено кластеров: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
print(f"Выбросы (шум): {sum(clusters == -1)} студентов")
print("\nРаспределение по кластерам:")
print(df['cluster'].value_counts().sort_index())

# Визуализация с помощью PCA (для 6D -> 2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_scaled)

plt.figure(figsize=(12, 5))

# Визуализация в 2D пространстве
plt.subplot(1, 2, 1)
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=df['cluster'], 
                     cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.title('DBSCAN кластеры (PCA проекция)')
plt.xlabel('PCA Компонента 1')
plt.ylabel('PCA Компонента 2')

# Визуализация выбросов
plt.subplot(1, 2, 2)
outliers = df[df['cluster'] == -1]
normal = df[df['cluster'] != -1]
plt.scatter(features_2d[df['cluster'] != -1, 0], 
           features_2d[df['cluster'] != -1, 1], 
           c=df[df['cluster'] != -1]['cluster'], cmap='viridis', alpha=0.7, label='Кластеры')
plt.scatter(features_2d[df['cluster'] == -1, 0], 
           features_2d[df['cluster'] == -1, 1], 
           c='red', marker='x', s=100, label='Выбросы')
plt.legend()
plt.title('Выбросы в кластеризации')
plt.xlabel('PCA Component 1')

plt.tight_layout()
plt.show()


#Визуализация в 3D пространстве
pca_3d = PCA(n_components=3)
features_3d = pca_3d.fit_transform(features_scaled)

from matplotlib.animation import FuncAnimation

def create_3d_animation():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], 
                        c=df['cluster'], cmap='tab10', s=60, alpha=0.7)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Анимация кластеров')
    
    def animate(frame):
        ax.view_init(elev=30, azim=frame)
        return scatter,
    
    anim = FuncAnimation(fig, animate, frames=range(0, 360, 2), 
                        interval=50, blit=False)
    
    plt.close()
    return anim

# anim = create_3d_animation()
# anim.save('clusters_3d.gif', writer='pillow', fps=20)

print("\n=== ХАРАКТЕРИСТИКИ КЛАСТЕРОВ ===")
cluster_stats = df.groupby('cluster').agg({
    'attempts': 'mean',
    'correctness': 'mean', 
    'time_spent_on_question': 'mean',
    'time_spent_on_material': 'mean',
    'selected_distractor_freq': 'mean',
    'study_time_preference': 'mean',
    'student_id': 'count'
}).round(2)

print(cluster_stats.rename(columns={'student_id': 'count'}))
