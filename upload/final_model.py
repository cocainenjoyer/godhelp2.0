import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
import joblib
import pickle
import pandas as pd
import json
import sys
import os


class MLRecommendationModel:

    
    def __init__(self):
        self.tfidf_genres = TfidfVectorizer(analyzer='word')
        self.tfidf_description = TfidfVectorizer(analyzer='word', max_features=5000)
        self.tfidf_countries = TfidfVectorizer(analyzer='word')
        self.scaler = MinMaxScaler()
        self.shows_df = None
        self.genres_matrix = None
        self.description_matrix = None
        self.countries_matrix = None
        self.ratings_matrix = None
        self.combined_features_matrix = None
        self.feature_weights = {
            'genres': 0.4,
            'description': 0.2,
            'countries': 0.1,
            'ratings': 0.3
        }
        
        # Иерархия возрастных пометок у сериалов по жесткости - сначала самые строгие
        self.age_rating_hierarchy = [
            "g", "tv-g", "tv-y",
            "tv-y7",
            "pg", "tv-pg",
            "tv-14",
            "r",
            "tv-ma"
        ]
        self.age_rating_strictness = {rating: i for i, rating in enumerate(self.age_rating_hierarchy)}
        
        # Карта соответствия настроения и жанров
        self.mood_to_genres_map = {
            "веселое": ["comedy", "animation", "family", "music"],
            "грустное": ["drama", "war", "history", "romance"],
            "печальное": ["drama", "romance", "war"],
            "меланхоличное": ["drama", "european", "romance"],
            "позитивное": ["comedy", "animation", "family", "music", "romance", "sport"],
            "трогательное": ["drama", "family", "animation", "romance", "documentation"],
            "радостное": ["comedy", "animation", "family", "music", "sport"],
            "расслабляющее": ["family", "animation", "documentation", "reality"],
            "захватывающее": ["action", "thriller", "scifi", "fantasy", "crime", "horror", "war", "sport"]
        }
        
        # Фильтр для компании по ее возрасту
        self.company_age_filters = {
            "один": {"allowed": [], "excluded": []},
            "с друзьями": {"allowed": ["tv-pg", "pg", "tv-14", "r", "tv-ma", "unknown"], "excluded": ["tv-y", "tv-g", "g", "tv-y7"]},
            "с семьёй": {"allowed": ["tv-y", "tv-g", "g", "tv-y7", "pg", "tv-pg", "tv-14", "unknown"], "excluded": ["r", "tv-ma"]},
            "с детьми": {"allowed": ["tv-y", "tv-g", "g", "tv-y7", "pg", "tv-pg"], "excluded": ["tv-14", "r", "tv-ma", "unknown"]}
        }
        
        # Рекомендации по жанрам, основываясь на компании (контингенте)
        self.company_genre_preferences = {
            "один": [],
            "с друзьями": ["action", "comedy", "horror", "thriller", "scifi", "fantasy", "sport", "crime"],
            "с семьёй": ["family", "comedy", "animation", "documentation", "history", "drama", "adventure", "music"],
            "с детьми": ["animation", "family", "adventure", "comedy", "music"]
        }
        
        # Жанры, которых следует избегать в зависимости от компании
        self.company_avoid_genres = {
            "один": [],
            "с друзьями": [],
            "с семьёй": ["horror"],
            "с детьми": ["horror", "thriller", "crime", "war", "drama", "romance", "action"]
        }
        
    def preprocess_data(self, df):
        self.shows_df=df.copy()
        # Преобразование списков в строки для TF-IDF
        self.shows_df['genres_str'] = self.shows_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        self.shows_df['countries_str'] = self.shows_df['production_countries'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # Нормализация текстовых полей
        self.shows_df['description'] = self.shows_df['description'].str.lower()
        self.shows_df['genres_str'] = self.shows_df['genres_str'].str.lower()
        self.shows_df['countries_str'] = self.shows_df['countries_str'].str.lower()
        
        # Создание матрицы рейтингов
        ratings = self.shows_df[['imdb_score', 'tmdb_score', 'tmdb_popularity']].values
        self.ratings_matrix = self.scaler.fit_transform(ratings)
        
        # Сбрасываем индексы для предотвращения проблем с индексацией
        self.shows_df = self.shows_df.reset_index(drop=True)
        
        return self.shows_df
    
    def fit(self, df=None):
        """
        Обучение модели на данных
        """
        if df is not None:
            self.preprocess_data(df)
        
        if self.shows_df is None:
            raise ValueError("No data available. Please provide a dataframe to fit() or call preprocess_data() first.")
        
        # Создание TF-IDF матриц
        self.genres_matrix = self.tfidf_genres.fit_transform(self.shows_df['genres_str'])
        self.description_matrix = self.tfidf_description.fit_transform(self.shows_df['description'])
        self.countries_matrix = self.tfidf_countries.fit_transform(self.shows_df['countries_str'])
        
        # Преобразуем матрицу рейтингов в разреженную матрицу для совместимости
        self.ratings_sparse_matrix = csr_matrix(self.ratings_matrix)
        
        # Комбинирование признаков с весами
        # Используем hstack для горизонтального объединения матриц разной размерности
        self.combined_features_matrix = hstack([
            self.feature_weights['genres'] * self.genres_matrix,
            self.feature_weights['description'] * self.description_matrix,
            self.feature_weights['countries'] * self.countries_matrix
        ])
        
        print(f"Model trained on {self.shows_df.shape[0]} shows")
        print(f"Feature matrices shapes: Genres {self.genres_matrix.shape}, Description {self.description_matrix.shape}, Countries {self.countries_matrix.shape}")
        print(f"Combined features matrix shape: {self.combined_features_matrix.shape}")
        
        return self
    
    def predict_similarity(self, show_idx):
        """
        Предсказание сходства между выбранным сериалом и всеми остальными
        """
        if self.combined_features_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Вычисление косинусного сходства
        similarity_scores = cosine_similarity(
            self.combined_features_matrix[show_idx:show_idx+1], 
            self.combined_features_matrix
        ).flatten()
        
        return similarity_scores
    
    def get_similar_shows(self, show_idx, n=10):
        """
        Получение n наиболее похожих сериалов
        """
        similarity_scores = self.predict_similarity(show_idx)
        
        # Исключаем сам сериал из результатов
        similarity_scores[show_idx] = 0
        
        # Получаем индексы топ-n наиболее похожих сериалов
        similar_indices = similarity_scores.argsort()[::-1][:n]
        
        # Формируем результат
        result = self.shows_df.iloc[similar_indices].copy()
        result['similarity_score'] = similarity_scores[similar_indices]
        
        return result
    
    def process_user_preferences(self, user_preferences):

        processed_prefs = {
            "genres": [],
            "release_year_min": None,
            "release_year_max": None,
            "runtime_min": None,
            "runtime_max": None,
            "production_country": None,
            "age_limit": None,
            "selected_mood": None,
            "selected_company": None
        }
        
        # Обработка жанров
        if 'preferred_genres' in user_preferences and user_preferences['preferred_genres']:
            processed_prefs["genres"] = [genre.lower() for genre in user_preferences['preferred_genres']]
        
        # Обработка временного периода
        if 'release_year_range' in user_preferences and user_preferences['release_year_range']:
            year_range = user_preferences['release_year_range']
            if year_range.endswith('-е'):
                start_year = int(year_range.split('-')[0])
                processed_prefs["release_year_min"] = start_year
                processed_prefs["release_year_max"] = start_year + 9
        
        # Обработка продолжительности серии
        if 'runtime_range' in user_preferences and user_preferences['runtime_range']:
            runtime_range = user_preferences['runtime_range']
            if runtime_range == 'до 60':
                processed_prefs["runtime_min"] = 0
                processed_prefs["runtime_max"] = 60
            elif runtime_range == '60-120':
                processed_prefs["runtime_min"] = 60
                processed_prefs["runtime_max"] = 120
            elif runtime_range == '120-180':
                processed_prefs["runtime_min"] = 120
                processed_prefs["runtime_max"] = 180
            elif runtime_range == '180-210':
                processed_prefs["runtime_min"] = 180
                processed_prefs["runtime_max"] = 210
        
        # Обработка страны производства
        if 'production_country' in user_preferences and user_preferences['production_country']:
            processed_prefs["production_country"] = user_preferences['production_country'].lower()
        
        # Обработка возрастного ограничения
        if 'age_limit' in user_preferences and user_preferences['age_limit']:
            processed_prefs["age_limit"] = user_preferences['age_limit'].lower()
        
        # Обработка настроения
        if 'mood' in user_preferences and user_preferences['mood']:
            processed_prefs["selected_mood"] = user_preferences['mood'].lower()
        
        # Обработка компании
        if 'company' in user_preferences and user_preferences['company']:
            processed_prefs["selected_company"] = user_preferences['company'].lower()
        
        return processed_prefs
    
    def get_recommendations(self, user_preferences, n=20):

        if self.combined_features_matrix is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Обработка пользовательских предпочтений
        processed_prefs = self.process_user_preferences(user_preferences)
        
        # Применяем фильтры
        filtered_df = self.shows_df.copy()
        
        # Фильтр по компании (возрастное ограничение)
        selected_company = processed_prefs.get("selected_company")
        if selected_company and selected_company in self.company_age_filters:
            age_filter_rules = self.company_age_filters[selected_company]
            if age_filter_rules.get("allowed") and len(age_filter_rules["allowed"]) > 0:
                filtered_df = filtered_df[filtered_df['age_certification'].str.lower().isin(age_filter_rules["allowed"])]
            elif age_filter_rules.get("excluded") and len(age_filter_rules["excluded"]) > 0:
                filtered_df = filtered_df[~filtered_df['age_certification'].str.lower().isin(age_filter_rules["excluded"])]
        
        # Фильтр по возрастному ограничению из анкеты (если более строгий)
        age_limit = processed_prefs.get("age_limit")
        if age_limit and age_limit != "unknown" and age_limit in self.age_rating_strictness:
            user_age_strictness = self.age_rating_strictness[age_limit]
            filtered_df = filtered_df[filtered_df['age_certification'].str.lower().apply(
                lambda x: self.age_rating_strictness.get(x, float('inf')) <= user_age_strictness if x in self.age_rating_strictness else True
            )]
        
        # Фильтр по жанрам
        if processed_prefs.get("genres") and len(processed_prefs["genres"]) > 0:
            preferred_genres_set = set(processed_prefs["genres"])
            filtered_df = filtered_df[filtered_df['genres'].apply(
                lambda x: len(preferred_genres_set.intersection(set([g.lower() for g in x]))) > 0 if isinstance(x, list) else False
            )]
        
        # Фильтр по годам выпуска
        if processed_prefs.get("release_year_min") is not None:
            filtered_df = filtered_df[filtered_df['release_year'] >= processed_prefs["release_year_min"]]
        if processed_prefs.get("release_year_max") is not None:
            filtered_df = filtered_df[filtered_df['release_year'] <= processed_prefs["release_year_max"]]
        
        # Фильтр по продолжительности
        if processed_prefs.get("runtime_min") is not None:
            filtered_df = filtered_df[filtered_df['runtime'] >= processed_prefs["runtime_min"]]
        if processed_prefs.get("runtime_max") is not None:
            filtered_df = filtered_df[filtered_df['runtime'] <= processed_prefs["runtime_max"]]
        
        # Фильтр по стране производства
        if processed_prefs.get("production_country"):
            country_pref = processed_prefs["production_country"].upper()
            filtered_df = filtered_df[filtered_df['production_countries'].apply(
                lambda x: country_pref in [c.upper() for c in x] if isinstance(x, list) else False
            )]
        
        # Если после фильтрации не осталось сериалов, возвращаем пустой датафрейм
        if filtered_df.empty:
            return pd.DataFrame()
        
        # Получаем индексы отфильтрованных сериалов
        filtered_indices = filtered_df.index.values
        
        # Проверяем, что индексы не выходят за границы массива
        valid_indices = [idx for idx in filtered_indices if idx < self.combined_features_matrix.shape[0]]
        
        if not valid_indices:
            return pd.DataFrame()
        
        # Создаем вектор пользовательских предпочтений по жанрам
        user_genres = ' '.join(processed_prefs.get('genres', []))
        user_genres_vector = self.tfidf_genres.transform([user_genres.lower()])
        
        # Если есть предпочтения по стране
        user_country = processed_prefs.get('production_country', '')
        user_country_vector = self.tfidf_countries.transform([user_country.lower() if user_country else ''])
        
        # Вычисляем сходство по жанрам и странам
        genre_similarity = cosine_similarity(user_genres_vector, self.genres_matrix).flatten()
        country_similarity = cosine_similarity(user_country_vector, self.countries_matrix).flatten()
        
        # Комбинируем сходство с весами
        combined_similarity = (
            0.7 * genre_similarity + 
            0.3 * country_similarity
        )
        
        # Вычисляем финальный скор для отфильтрованных сериалов
        filtered_similarity = combined_similarity[valid_indices]
        
        # Получаем рейтинги для отфильтрованных сериалов
        filtered_ratings = self.ratings_matrix[valid_indices]
        
        # Нормализуем рейтинги
        normalized_ratings = filtered_ratings.mean(axis=1)
        
        # Вычисляем дополнительные скоры для настроения и компании
        mood_score = np.zeros(len(valid_indices))
        company_score = np.zeros(len(valid_indices))
        
        # Скор по настроению
        selected_mood = processed_prefs.get("selected_mood")
        if selected_mood and selected_mood in self.mood_to_genres_map:
            mood_genres = set(self.mood_to_genres_map[selected_mood])
            for i, idx in enumerate(valid_indices):
                show_genres = set([g.lower() for g in self.shows_df.iloc[idx]['genres']])
                mood_match = len(mood_genres.intersection(show_genres)) / len(mood_genres) if len(mood_genres) > 0 else 0
                mood_score[i] = mood_match
        
        # Скор по компании
        selected_company = processed_prefs.get("selected_company")
        if selected_company and selected_company in self.company_genre_preferences:
            company_genres = set(self.company_genre_preferences[selected_company])
            avoid_genres = set(self.company_avoid_genres[selected_company])
            
            for i, idx in enumerate(valid_indices):
                show_genres = set([g.lower() for g in self.shows_df.iloc[idx]['genres']])
                
                # Положительный скор за предпочтительные жанры
                if len(company_genres) > 0:
                    company_match = len(company_genres.intersection(show_genres)) / len(company_genres)
                else:
                    company_match = 0
                
                # Отрицательный скор за нежелательные жанры
                if len(avoid_genres) > 0:
                    avoid_match = len(avoid_genres.intersection(show_genres)) / len(avoid_genres)
                else:
                    avoid_match = 0
                
                company_score[i] = company_match - avoid_match
        
        # Комбинируем все скоры
        final_scores = (
            0.5 * filtered_similarity + 
            0.2 * normalized_ratings + 
            0.15 * mood_score + 
            0.15 * company_score
        )
        
        # Сортируем по финальному скору
        sorted_indices = np.argsort(final_scores)[::-1][:n]
        
        # Получаем индексы в исходном датафрейме
        result_indices = [valid_indices[i] for i in sorted_indices]
        
        # Формируем результат
        result = self.shows_df.iloc[result_indices].copy()
        result['similarity_score'] = filtered_similarity[sorted_indices]
        result['rating_score'] = normalized_ratings[sorted_indices]
        result['mood_score'] = mood_score[sorted_indices]
        result['company_score'] = company_score[sorted_indices]
        result['final_score'] = final_scores[sorted_indices]
        
        return result[['title', 'genres', 'release_year', 'age_certification', 'imdb_score', 'runtime', 'seasons', 
                      'similarity_score', 'rating_score', 'mood_score', 'company_score', 'final_score']]
    
    def save_model(self, filepath):

        model_data = {
            'tfidf_genres': self.tfidf_genres,
            'tfidf_description': self.tfidf_description,
            'tfidf_countries': self.tfidf_countries,
            'scaler': self.scaler,
            'genres_matrix': self.genres_matrix,
            'description_matrix': self.description_matrix,
            'countries_matrix': self.countries_matrix,
            'ratings_matrix': self.ratings_matrix,
            'combined_features_matrix': self.combined_features_matrix,
            'feature_weights': self.feature_weights,
            'age_rating_hierarchy': self.age_rating_hierarchy,
            'age_rating_strictness': self.age_rating_strictness,
            'mood_to_genres_map': self.mood_to_genres_map,
            'company_age_filters': self.company_age_filters,
            'company_genre_preferences': self.company_genre_preferences,
            'company_avoid_genres': self.company_avoid_genres
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):

        model_data = joblib.load(filepath)
        
        self.tfidf_genres = model_data['tfidf_genres']
        self.tfidf_description = model_data['tfidf_description']
        self.tfidf_countries = model_data['tfidf_countries']
        self.scaler = model_data['scaler']
        self.genres_matrix = model_data['genres_matrix']
        self.description_matrix = model_data['description_matrix']
        self.countries_matrix = model_data['countries_matrix']
        self.ratings_matrix = model_data['ratings_matrix']
        self.combined_features_matrix = model_data['combined_features_matrix']
        self.feature_weights = model_data['feature_weights']
        
        # Загрузка дополнительных данных, если они есть
        if 'age_rating_hierarchy' in model_data:
            self.age_rating_hierarchy = model_data['age_rating_hierarchy']
        if 'age_rating_strictness' in model_data:
            self.age_rating_strictness = model_data['age_rating_strictness']
        if 'mood_to_genres_map' in model_data:
            self.mood_to_genres_map = model_data['mood_to_genres_map']
        if 'company_age_filters' in model_data:
            self.company_age_filters = model_data['company_age_filters']
        if 'company_genre_preferences' in model_data:
            self.company_genre_preferences = model_data['company_genre_preferences']
        if 'company_avoid_genres' in model_data:
            self.company_avoid_genres = model_data['company_avoid_genres']
        
        print(f"Model loaded from {filepath}")
        
    def evaluate(self, test_size=0.2, random_state=42):

        if self.shows_df is None:
            raise ValueError("No data available for evaluation")
            
        # Создаем копию данных для оценки
        eval_df = self.shows_df.copy().reset_index(drop=True)
        
        # Разделяем данные на обучающую и тестовую выборки
        train_indices, test_indices = train_test_split(
            range(len(eval_df)), 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Создаем отдельную модель для оценки
        eval_model = MLRecommendationModel()
        eval_model.fit(eval_df.iloc[train_indices])
        
        # Метрики для оценки
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        
        # Ограничиваем количество тестовых примеров для ускорения
        test_sample = min(100, len(test_indices))
        
        # Для каждого сериала в тестовой выборке
        for i in range(test_sample):
            idx = test_indices[i]
            row = eval_df.iloc[idx]
            
            # Получаем жанры текущего сериала
            true_genres = set(row['genres'])
            
            if not true_genres:  # Пропускаем, если нет жанров
                continue
                
            # Получаем рекомендации на основе жанров
            user_prefs = {'preferred_genres': list(true_genres)}
            recommendations = eval_model.get_recommendations(user_prefs, n=10)
            
            if recommendations.empty:
                continue
            
            # Вычисляем метрики
            # Precision@k: доля релевантных рекомендаций среди всех рекомендаций
            relevant_count = 0
            for _, rec_row in recommendations.iterrows():
                rec_genres = set(rec_row['genres'])
                if len(true_genres.intersection(rec_genres)) > 0:
                    relevant_count += 1
            
            precision = relevant_count / len(recommendations)
            precision_at_k.append(precision)
            
            # Recall@k: доля релевантных рекомендаций среди всех релевантных сериалов
            # Для упрощения, считаем все сериалы с хотя бы одним общим жанром релевантными
            all_relevant = len([1 for _, df_row in eval_model.shows_df.iterrows() 
                              if len(true_genres.intersection(set(df_row['genres']))) > 0])
            
            recall = relevant_count / all_relevant if all_relevant > 0 else 0
            recall_at_k.append(recall)
            
            # NDCG@k: нормализованный дисконтированный кумулятивный выигрыш
            # Учитывает порядок рекомендаций
            dcg = 0
            idcg = 0
            
            for i, (_, rec_row) in enumerate(recommendations.iterrows()):
                rec_genres = set(rec_row['genres'])
                relevance = len(true_genres.intersection(rec_genres)) / len(true_genres) if len(true_genres) > 0 else 0
                dcg += relevance / np.log2(i + 2)  # i+2 потому что i начинается с 0
            
            # Идеальный DCG (если бы все рекомендации были идеально релевантными)
            for i in range(len(recommendations)):
                idcg += 1 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k.append(ndcg)
        
        # Вычисляем средние значения метрик
        avg_precision = np.mean(precision_at_k) if precision_at_k else 0
        avg_recall = np.mean(recall_at_k) if recall_at_k else 0
        avg_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0
        
        # Возвращаем результаты
        return {
            'Precision@10': avg_precision,
            'Recall@10': avg_recall,
            'NDCG@10': avg_ndcg
        }

    def load_user_preferences_from_json(self, json_file_path):
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                user_preferences = json.load(f)
            print(f"Пользовательские предпочтения успешно загружены из {json_file_path}")
            return user_preferences
        except Exception as e:
            print(f"Ошибка при загрузке пользовательских предпочтений из {json_file_path}: {e}")
            return None
        
    def save_recommendations_to_json(self, recommendations, json_file_path):
      
        

        recommendations_list = []
        for _, row in recommendations.iterrows():
            row_dict = {
                    "title": row["title"],
                    "genres": row["genres"],
                    "description": self.shows_df.loc[self.shows_df["title"] == row["title"], "description"].values[0] if "description" in self.shows_df.columns else "",
                    "release_year": int(row["release_year"]) if isinstance(row["release_year"], (np.int64, np.int32, np.int16, np.int8)) else row["release_year"],
                    "imdb_score": float(row["imdb_score"]) if isinstance(row["imdb_score"], (np.float64, np.float32, np.float16)) else row["imdb_score"]}

            recommendations_list.append(row_dict)
        
        result = {
            "recommendations": recommendations_list
        }
    
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"Рекомендации успешно сохранены в {json_file_path}")
            return True
        except Exception as e:
            print(f"Ошибка при сохранении рекомендаций в {json_file_path}: {e}")
            return False

    def process_json_input_output(self, input_json_path, output_json_path, num_recommendations=10):
        
        user_preferences = self.load_user_preferences_from_json(input_json_path)
        try:
            recommendations = self.get_recommendations(user_preferences, n=num_recommendations)

            return self.save_recommendations_to_json(recommendations, output_json_path)
        except Exception as e:
            print(f"ошибка {e}")
            return False






with open("fixed_dataset.pkl", "rb") as f:
    data = pickle.load(f)

    
dataset_path = sys.argv[1]
input_json_path = sys.argv[2]
output_json_path = sys.argv[3]
    

model = MLRecommendationModel()
model.fit(data)
result = model.process_json_input_output(input_json_path, output_json_path)