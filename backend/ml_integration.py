import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import requests

# Добавляем путь к моделям в sys.path
sys.path.append('/app/data')

# Импортируем модели машинного обучения
try:
    from final_model import MLRecommendationModel
except ImportError:
    print("Warning: Could not import MLRecommendationModel")

class MLIntegration:
    """
    Класс для интеграции моделей машинного обучения с основным сервисом
    """
    
    def __init__(self):
        """
        Инициализация интеграции с ML-моделями
        """
        self.data_path = os.getenv("DATA_PATH", "/app/data/data_final.pkl")
        self.desc_embeddings_path = os.getenv("DESC_EMBEDDINGS_PATH", "/app/data/description_embeddings.npy")
        self.keywords_embeddings_path = os.getenv("KEYWORDS_EMBEDDINGS_PATH", "/app/data/keywords_embeddings.npy")
        self.tmdb_api_key = os.getenv("TMDB_API_KEY", "your-tmdb-api-key")
        self.poster_base_url = "https://image.tmdb.org/t/p/w500"
        
        # Загружаем данные и модели
        self.data = None
        self.description_embeddings = None
        self.keywords_embeddings = None
        self.profile_model = None
        
        try:
            self.load_data_and_models()
        except Exception as e:
            print(f"Error loading data and models: {e}")
    
    def load_data_and_models(self):
        """
        Загрузка данных и моделей машинного обучения
        """
        # Загружаем датасет
        try:
            self.data = pd.read_pickle(self.data_path)
            print(f"Dataset loaded: {len(self.data)} shows")
        except Exception as e:
            print(f"Error loading dataset: {e}")
        
        # Загружаем эмбеддинги описаний
        try:
            self.description_embeddings = np.load(self.desc_embeddings_path)
            print(f"Description embeddings loaded: {self.description_embeddings.shape}")
        except Exception as e:
            print(f"Error loading description embeddings: {e}")
        
        # Загружаем эмбеддинги ключевых слов
        try:
            self.keywords_embeddings = np.load(self.keywords_embeddings_path)
            print(f"Keywords embeddings loaded: {self.keywords_embeddings.shape}")
        except Exception as e:
            print(f"Error loading keywords embeddings: {e}")
        
        # Инициализируем модель для рекомендаций по профилю
        try:
            self.profile_model = MLRecommendationModel()
            self.profile_model.preprocess_data(self.data)
            self.profile_model.fit()
            print("Profile recommendation model initialized")
        except Exception as e:
            print(f"Error initializing profile model: {e}")
    
    def get_poster_url(self, title: str) -> str:
        """
        Получение URL постера для сериала по названию
        
        :param title: Название сериала
        :return: URL постера или заглушка, если постер не найден
        """
        try:
            # Проверяем, есть ли сериал в датасете
            if self.data is not None:
                show_data = self.data[self.data['title'] == title]
                if not show_data.empty and 'poster_path' in show_data.columns:
                    poster_path = show_data['poster_path'].iloc[0]
                    if poster_path and not pd.isna(poster_path):
                        return f"{self.poster_base_url}{poster_path}"
            
            # Если постер не найден в датасете, запрашиваем через API
            search_url = f"https://api.themoviedb.org/3/search/tv"
            params = {
                "api_key": self.tmdb_api_key,
                "query": title,
                "language": "ru-RU"
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    return f"{self.poster_base_url}{poster_path}"
        except Exception as e:
            print(f"Error getting poster URL: {e}")
        
        # Если не удалось получить постер, возвращаем заглушку
        return "https://via.placeholder.com/200?text=No+Poster"
    
    def get_similar_shows(self, show_title: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Получение похожих сериалов по названию
        
        :param show_title: Название сериала
        :param top_n: Количество рекомендаций
        :return: Список похожих сериалов
        """
        if self.data is None or self.description_embeddings is None or self.keywords_embeddings is None:
            return []
        
        try:
            # Создаем временный JSON-файл для входных данных
            input_data = {"title": show_title}
            with open('input.json', 'w', encoding='utf-8') as f:
                json.dump(input_data, f, ensure_ascii=False)
            
            # Импортируем функцию из модуля similar_prod
            sys.path.append('/app/data')
            from similar_prod import load_recommendation_model, get_recommendations
            
            # Получаем рекомендации
            data, desc_embeddings, keywords_embeddings = load_recommendation_model(
                self.data_path, self.desc_embeddings_path, self.keywords_embeddings_path
            )
            
            recommendations = get_recommendations(
                data, desc_embeddings, keywords_embeddings,
                show_title, top_n=top_n
            )
            
            # Проверяем, является ли результат строкой (сообщение об ошибке)
            if isinstance(recommendations, str):
                print(f"Error getting recommendations: {recommendations}")
                return []
            
            # Преобразуем рекомендации в нужный формат
            result = []
            for _, row in recommendations.iterrows():
                show_id = int(row.name) if hasattr(row, 'name') else 0
                title = row.get('title', '')
                
                # Получаем URL постера
                poster_url = self.get_poster_url(title)
                
                show = {
                    "id": show_id,
                    "title": title,
                    "description": row.get('description', ''),
                    "image": poster_url,  # Используем реальный URL постера
                    "rating": float(row.get('imdb_score', 0)),
                    "year": int(row.get('release_year', 0)),
                    "genres": row.get('genres', [])
                }
                result.append(show)
            
            return result
        except Exception as e:
            print(f"Error getting similar shows: {e}")
            return []
    
    def get_recommendations_by_profile(self, user_preferences: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Получение рекомендаций на основе профиля пользователя
        
        :param user_preferences: Предпочтения пользователя
        :param top_n: Количество рекомендаций
        :return: Список рекомендованных сериалов
        """
        if self.profile_model is None or self.data is None:
            return []
        
        try:
            # Получаем рекомендации от модели
            recommendations = self.profile_model.get_recommendations(user_preferences, n=top_n)
            
            # Если рекомендаций нет, возвращаем пустой список
            if recommendations.empty:
                return []
            
            # Преобразуем рекомендации в нужный формат
            result = []
            for _, row in recommendations.iterrows():
                show_id = int(row.name) if hasattr(row, 'name') else 0
                title = row.get('title', '')
                
                # Получаем URL постера
                poster_url = self.get_poster_url(title)
                
                show = {
                    "id": show_id,
                    "title": title,
                    "description": row.get('description', ''),
                    "image": poster_url,  # Используем реальный URL постера
                    "rating": float(row.get('imdb_score', 0)),
                    "year": int(row.get('release_year', 0)),
                    "genres": row.get('genres', [])
                }
                result.append(show)
            
            return result
        except Exception as e:
            print(f"Error getting recommendations by profile: {e}")
            return []
    
    def get_recommendations_by_quiz(self, quiz_answers: List[Dict[str, str]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Получение рекомендаций на основе ответов на анкету
        
        :param quiz_answers: Ответы на анкету
        :param top_n: Количество рекомендаций
        :return: Список рекомендованных сериалов
        """
        if self.profile_model is None or self.data is None:
            return []
        
        try:
            # Преобразуем ответы на анкету в формат, понятный модели
            user_preferences = {
                "mood": None,
                "company": None,
                "production_country": None
            }
            
            for answer in quiz_answers:
                if answer.get("id") == "mood":
                    user_preferences["mood"] = answer.get("option")
                elif answer.get("id") == "company":
                    user_preferences["company"] = answer.get("option")
                elif answer.get("id") == "country":
                    user_preferences["production_country"] = answer.get("option")
            
            # Получаем рекомендации от модели
            recommendations = self.profile_model.get_recommendations(user_preferences, n=top_n)
            
            # Если рекомендаций нет, возвращаем пустой список
            if recommendations.empty:
                return []
            
            # Преобразуем рекомендации в нужный формат
            result = []
            for _, row in recommendations.iterrows():
                show_id = int(row.name) if hasattr(row, 'name') else 0
                title = row.get('title', '')
                
                # Получаем URL постера
                poster_url = self.get_poster_url(title)
                
                show = {
                    "id": show_id,
                    "title": title,
                    "description": row.get('description', ''),
                    "image": poster_url,  # Используем реальный URL постера
                    "rating": float(row.get('imdb_score', 0)),
                    "year": int(row.get('release_year', 0)),
                    "genres": row.get('genres', [])
                }
                result.append(show)
            
            return result
        except Exception as e:
            print(f"Error getting recommendations by quiz: {e}")
            return []
    
    def import_shows_from_dataset(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Импорт сериалов из датасета в базу данных
        
        :param limit: Максимальное количество сериалов для импорта
        :return: Список импортированных сериалов
        """
        if self.data is None:
            return []
        
        try:
            # Берем первые limit записей из датасета
            shows_to_import = self.data.head(limit)
            
            # Преобразуем в нужный формат
            result = []
            for _, row in shows_to_import.iterrows():
                show_id = int(row.name) if hasattr(row, 'name') else 0
                title = row.get('title', '')
                
                # Получаем URL постера
                poster_url = self.get_poster_url(title)
                
                show = {
                    "id": show_id,
                    "title": title,
                    "description": row.get('description', ''),
                    "release_year": int(row.get('release_year', 0)),
                    "genres": row.get('genres', []),
                    "runtime": int(row.get('runtime', 0)),
                    "imdb_score": float(row.get('imdb_score', 0)),
                    "tmdb_score": float(row.get('tmdb_score', 0)),
                    "tmdb_popularity": float(row.get('tmdb_popularity', 0)),
                    "age_certification": row.get('age_certification', ''),
                    "production_countries": row.get('production_countries', []),
                    "keywords": row.get('keywords', []),
                    "poster_url": poster_url  # Добавляем URL постера
                }
                result.append(show)
            
            return result
        except Exception as e:
            print(f"Error importing shows from dataset: {e}")
            return []
