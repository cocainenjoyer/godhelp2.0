import os
import requests
from typing import Optional, Dict, Any, List

class PosterAPI:

    def __init__(self, api_key: str = None):
        """
        Инициализация API клиента
        
        :param api_key: API ключ для TMDB (The Movie Database)
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY", "your-tmdb-api-key")
        self.base_url = "https://api.themoviedb.org/3"
        self.poster_base_url = "https://image.tmdb.org/t/p/w500"
        self.storage_path = os.getenv("POSTER_STORAGE_PATH", "./posters")
        
        # Создаем директорию для хранения постеров, если она не существует
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def search_show(self, title: str, language: str = "ru-RU") -> Dict[str, Any]:
        """
        Поиск сериала по названию
        
        :param title: Название сериала
        :param language: Язык результатов (по умолчанию русский)
        :return: Информация о найденном сериале или пустой словарь
        """
        try:
            search_url = f"{self.base_url}/search/tv"
            params = {
                "api_key": self.api_key,
                "query": title,
                "language": language
            }
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            results = response.json().get("results", [])
            
            if results:
                return results[0]
            return {}
        except requests.RequestException as e:
            print(f"Error searching for show: {e}")
            return {}
    
    def get_poster_url(self, title: str, language: str = "ru-RU") -> Optional[str]:
        """
        Получение URL постера сериала по названию
        
        :param title: Название сериала
        :param language: Язык результатов (по умолчанию русский)
        :return: URL постера или None, если постер не найден
        """
        show_info = self.search_show(title, language)
        poster_path = show_info.get("poster_path")
        
        if poster_path:
            return f"{self.poster_base_url}{poster_path}"
        return None
    
    def save_poster(self, title: str, show_id: int, language: str = "ru-RU") -> Optional[str]:
        """
        Сохранение постера сериала локально
        
        :param title: Название сериала
        :param show_id: ID сериала в базе данных
        :param language: Язык результатов (по умолчанию русский)
        :return: Путь к сохраненному файлу или None в случае ошибки
        """
        poster_url = self.get_poster_url(title, language)
        if not poster_url:
            return None
        
        try:
            response = requests.get(poster_url)
            response.raise_for_status()
            
            file_extension = os.path.splitext(poster_url)[1] or ".jpg"
            file_path = os.path.join(self.storage_path, f"show_{show_id}{file_extension}")
            
            with open(file_path, "wb") as f:
                f.write(response.content)
                
            return file_path
        except Exception as e:
            print(f"Error saving poster: {e}")
            return None
    
    def get_show_details(self, show_id: int, language: str = "ru-RU") -> Dict[str, Any]:
        """
        Получение детальной информации о сериале по его TMDB ID
        
        :param show_id: ID сериала в TMDB
        :param language: Язык результатов (по умолчанию русский)
        :return: Детальная информация о сериале
        """
        try:
            details_url = f"{self.base_url}/tv/{show_id}"
            params = {
                "api_key": self.api_key,
                "language": language
            }
            response = requests.get(details_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting show details: {e}")
            return {}
    
    def get_popular_shows(self, page: int = 1, language: str = "ru-RU") -> List[Dict[str, Any]]:
        """
        Получение списка популярных сериалов
        
        :param page: Номер страницы результатов
        :param language: Язык результатов (по умолчанию русский)
        :return: Список популярных сериалов
        """
        try:
            popular_url = f"{self.base_url}/tv/popular"
            params = {
                "api_key": self.api_key,
                "language": language,
                "page": page
            }
            response = requests.get(popular_url, params=params)
            response.raise_for_status()
            return response.json().get("results", [])
        except requests.RequestException as e:
            print(f"Error getting popular shows: {e}")
            return []
    
    def get_show_by_tmdb_id(self, tmdb_id: int, language: str = "ru-RU") -> Dict[str, Any]:
        """
        Получение информации о сериале по его TMDB ID
        
        :param tmdb_id: ID сериала в TMDB
        :param language: Язык результатов (по умолчанию русский)
        :return: Информация о сериале
        """
        try:
            show_url = f"{self.base_url}/tv/{tmdb_id}"
            params = {
                "api_key": self.api_key,
                "language": language
            }
            response = requests.get(show_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error getting show by TMDB ID: {e}")
            return {}
