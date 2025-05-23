import requests
import json
import os
import time

# Базовый URL API
BASE_URL = "http://localhost:8000"

def test_auth_endpoints():
    """Тестирование эндпоинтов аутентификации"""
    print("\n=== Тестирование аутентификации ===")
    
    # Регистрация нового пользователя
    register_data = {
        "email": f"test_user_{int(time.time())}@example.com",
        "username": f"test_user_{int(time.time())}",
        "password": "test_password"
    }
    
    print(f"Регистрация пользователя: {register_data['username']}")
    response = requests.post(f"{BASE_URL}/auth/register", json=register_data)
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        token_data = response.json()
        print(f"Токен получен: {token_data['access_token'][:20]}...")
        
        # Сохраняем токен для дальнейших запросов
        auth_header = {"Authorization": f"Bearer {token_data['access_token']}"}
        
        # Логин с теми же данными
        login_data = {
            "email": register_data["email"],
            "password": register_data["password"]
        }
        
        print(f"\nЛогин пользователя: {login_data['email']}")
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            token_data = response.json()
            print(f"Токен получен: {token_data['access_token'][:20]}...")
            return token_data['access_token']
    
    return None

def test_initial_preferences(user_id=1):
    """Тестирование регистрации начальных предпочтений"""
    print("\n=== Тестирование начальных предпочтений ===")
    
    # Данные для начальных предпочтений
    initial_data = {
        "id": user_id,
        "email": f"user_{user_id}@example.com",
        "username": f"user_{user_id}",
        "initial": {
            "genres": ["drama", "comedy"],
            "period": "2010-е",
            "duration": "60-120"
        }
    }
    
    print(f"Регистрация начальных предпочтений для пользователя {user_id}")
    response = requests.post(f"{BASE_URL}/users/register-initial", json=initial_data)
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Результат: {result}")
        return True
    
    return False

def test_recommendations(user_id=1):
    """Тестирование рекомендаций"""
    print("\n=== Тестирование рекомендаций ===")
    
    # Получение рекомендаций по профилю
    print(f"Получение рекомендаций по профилю для пользователя {user_id}")
    response = requests.get(f"{BASE_URL}/recommendations/profile?id={user_id}")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        recommendations = response.json()
        print(f"Получено рекомендаций: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:2]):  # Показываем только первые 2
            print(f"  {i+1}. {rec['title']} ({rec['year']}) - {rec['rating']}")
    
    # Тестирование рекомендаций по анкете
    quiz_data = {
        "data": [
            {"id": "mood", "option": "веселое"},
            {"id": "country", "option": "США"},
            {"id": "company", "option": "с друзьями"}
        ]
    }
    
    print(f"\nПолучение рекомендаций по анкете для пользователя {user_id}")
    response = requests.post(f"{BASE_URL}/recommendations/by-quiz?user_id={user_id}", json=quiz_data)
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        recommendations = response.json()
        print(f"Получено рекомендаций: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:2]):  # Показываем только первые 2
            print(f"  {i+1}. {rec['title']} ({rec['year']}) - {rec['rating']}")
    
    # Тестирование похожих сериалов
    show_id = 1  # ID сериала для поиска похожих
    print(f"\nПоиск похожих сериалов для ID {show_id}")
    response = requests.get(f"{BASE_URL}/recommendations/similar-by-id?id={show_id}&user_id={user_id}")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        recommendations = response.json()
        print(f"Получено похожих сериалов: {len(recommendations)}")
        for i, rec in enumerate(recommendations[:2]):  # Показываем только первые 2
            print(f"  {i+1}. {rec['title']} ({rec['year']}) - {rec['rating']}")

def test_posters():
    """Тестирование API постеров"""
    print("\n=== Тестирование API постеров ===")
    
    # Инициализация демо-сериалов
    print("Инициализация демо-сериалов")
    response = requests.post(f"{BASE_URL}/titles/init-demo")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Результат: {result}")
    
    # Получение карты сериалов
    print("\nПолучение карты сериалов")
    response = requests.get(f"{BASE_URL}/titles/map")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        shows = response.json()
        print(f"Получено сериалов: {len(shows)}")
        
        if shows:
            # Выбираем первый сериал для тестирования постеров
            show = shows[0]
            show_id = show["id"]
            title = show["title"]
            
            # Загрузка постера по названию
            print(f"\nЗагрузка постера для сериала '{title}'")
            response = requests.post(f"{BASE_URL}/posters/fetch-by-title?title={title}")
            print(f"Статус: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Результат: {result}")
            
            # Получение информации о постере
            print(f"\nПолучение информации о постере для сериала ID {show_id}")
            response = requests.get(f"{BASE_URL}/posters/info/{show_id}")
            print(f"Статус: {response.status_code}")
            
            if response.status_code == 200:
                poster_info = response.json()
                print(f"Информация о постере: {poster_info.get('poster_url', 'Нет URL')}")

def test_history(user_id=1):
    """Тестирование истории пользователя"""
    print("\n=== Тестирование истории пользователя ===")
    
    # Получение истории пользователя
    print(f"Получение истории для пользователя {user_id}")
    response = requests.get(f"{BASE_URL}/users/{user_id}/history")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        history = response.json()
        print(f"Записей в истории: {len(history.get('history', []))}")
    
    # Добавление записи в историю
    show_id = 1  # ID сериала для добавления в историю
    interaction_type = "view"
    
    print(f"\nДобавление записи в историю: пользователь {user_id}, сериал {show_id}, тип {interaction_type}")
    response = requests.post(f"{BASE_URL}/users/{user_id}/history?show_id={show_id}&interaction_type={interaction_type}")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Результат: {result}")
        
        # Проверяем, что запись добавилась
        print(f"\nПроверка добавления записи в историю")
        response = requests.get(f"{BASE_URL}/users/{user_id}/history")
        print(f"Статус: {response.status_code}")
        
        if response.status_code == 200:
            history = response.json()
            print(f"Записей в истории: {len(history.get('history', []))}")

def test_ml_integration():
    """Тестирование интеграции с ML-моделями"""
    print("\n=== Тестирование интеграции с ML-моделями ===")
    
    # Импорт сериалов из датасета
    print("Импорт сериалов из датасета (ограничение: 10)")
    response = requests.post(f"{BASE_URL}/recommendations/import-shows?limit=10")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Результат: {result}")

def test_debug_endpoints():
    """Тестирование отладочных эндпоинтов"""
    print("\n=== Тестирование отладочных эндпоинтов ===")
    
    # Получение отладочной информации о базе данных
    print("Получение отладочной информации о базе данных")
    response = requests.get(f"{BASE_URL}/debug/db")
    print(f"Статус: {response.status_code}")
    
    if response.status_code == 200:
        debug_info = response.json()
        print(f"Пользователей: {len(debug_info.get('users', []))}")
        print(f"Предпочтений: {len(debug_info.get('initial_preferences', []))}")
        print(f"Сериалов: {len(debug_info.get('shows', []))}")
        print(f"Записей истории: {len(debug_info.get('history', []))}")
        print(f"Рекомендаций: {len(debug_info.get('recommendations', []))}")

def run_all_tests():
    """Запуск всех тестов"""
    print("=== Запуск всех тестов API ===")
    
    # Тестирование аутентификации
    token = test_auth_endpoints()
    
    # Тестирование начальных предпочтений
    test_initial_preferences()
    
    # Тестирование рекомендаций
    test_recommendations()
    
    # Тестирование постеров
    test_posters()
    
    # Тестирование истории
    test_history()
    
    # Тестирование интеграции с ML-моделями
    test_ml_integration()
    
    # Тестирование отладочных эндпоинтов
    test_debug_endpoints()
    
    print("\n=== Тестирование завершено ===")

if __name__ == "__main__":
    run_all_tests()
