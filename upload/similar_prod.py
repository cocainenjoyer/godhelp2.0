from fuzzywuzzy import process
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

with open('input.json', 'r') as file:
       json_data = json.load(file)

def load_recommendation_model(data_path, desc_embeddings_path, keywords_embeddings_path):
    """
    Загружает предварительно обработанные данные и эмбеддинги для рекомендательной системы.

    Параметры:
    data_path (str): Путь к файлу с обработанным датафреймом
    desc_embeddings_path (str): Путь к файлу с эмбеддингами описаний
    keywords_embeddings_path (str): Путь к файлу с эмбеддингами ключевых слов

    Возвращает:
    tuple: (обработанный датафрейм, эмбеддинги описаний, эмбеддинги ключевых слов)
    """

    with open('input.json', 'r') as file:
        input = json.load(file)

    # Загружаем обработанный датафрейм
    data = pd.read_pickle(data_path)

    # Загружаем эмбеддинги описаний
    description_embeddings = np.load(desc_embeddings_path)

    # Загружаем эмбеддинги ключевых слов
    keywords_embeddings = np.load(keywords_embeddings_path)

    return data, description_embeddings, keywords_embeddings

def get_recommendations(data, description_embeddings, keywords_embeddings,
                       show_title, top_n=5, keywords_weight=0.3, respect_animation_preference=True):
    """
    Находит сериалы, похожие на указанный, используя предварительно обработанные данные.

    Параметры:
    data (DataFrame): Обработанный датафрейм с данными о сериалах
    description_embeddings (numpy.ndarray): Матрица эмбеддингов описаний
    keywords_embeddings (numpy.ndarray): Матрица эмбеддингов ключевых слов
    show_title (str): Название целевого сериала
    top_n (int): Количество рекомендаций
    keywords_weight (float): Вес ключевых слов в итоговой оценке (от 0 до 1)
    respect_animation_preference (bool): Учитывать ли предпочтения по анимации

    Возвращает:
    DataFrame: Датафрейм с рекомендованными сериалами
    """
    # Нечеткий поиск по названию
    # Фильтруем только строковые значения для поиска
    titles = []
    title_indices = []

    for i, title in enumerate(data['title']):
        if isinstance(title, str):
            titles.append(title)
            title_indices.append(i)

    if not titles:
        return "В базе данных нет корректных названий сериалов."

    # Сначала проверяем точное совпадение (без учета регистра)
    exact_matches = [title for title in titles if title.lower() == show_title.lower()]
    if exact_matches:
        matched_title = exact_matches[0]
        match_score = 100
    else:
        # Если точного совпадения нет, используем нечеткий поиск
        best_match = process.extractOne(show_title, titles)
        matched_title = best_match[0]
        match_score = best_match[1]

    if match_score < 87:  # Порог сходства
        return f"Сериал с названием '{show_title}' не найден в базе данных."

    # Получаем индекс найденного сериала
    idx = data[data['title'] == matched_title].index[0]

    # Получаем жанры целевого сериала
    target_genres = set(data.loc[idx, 'genres'])

    is_target_animation = any(genre.lower() in ['animation', 'anime', 'cartoon'] for genre in target_genres)

    if not target_genres:
        return f"У сериала '{matched_title}' не указаны жанры. Невозможно найти похожие сериалы."

    # Этап 1: Отбираем сериалы, которые содержат ВСЕ жанры целевого сериала
    exact_match_indices = []
    partial_match_indices = []

    for i, row in data.iterrows():
        if i == idx:  # Пропускаем исходный сериал
            continue

        show_genres = set(row['genres'])

        if respect_animation_preference:
            is_show_animation = any(genre.lower() in ['animation', 'anime', 'cartoon'] for genre in show_genres)

            # Если целевой сериал не анимационный, а текущий анимационный - пропускаем
            if not is_target_animation and is_show_animation:
                continue

        # Проверяем, содержит ли сериал все жанры целевого сериала
        if target_genres.issubset(show_genres):
            exact_match_indices.append(i)
        # Также сохраняем сериалы с частичным совпадением жанров для запасного варианта
        elif target_genres.intersection(show_genres):
            match_ratio = len(target_genres.intersection(show_genres)) / len(target_genres)
            partial_match_indices.append((i, match_ratio))

    # Если нашли сериалы с полным совпадением жанров
    if exact_match_indices:
        candidate_indices = exact_match_indices

        # Если сериалов с полным совпадением меньше, чем top_n, добавляем сериалы с частичным совпадением
        if len(exact_match_indices) < top_n and partial_match_indices:
            # Сортируем по доле совпадающих жанров
            partial_match_indices.sort(key=lambda x: x[1], reverse=True)
            # Добавляем лучшие частичные совпадения
            additional_indices = [i for i, _ in partial_match_indices[:min(top_n - len(exact_match_indices), len(partial_match_indices))]]
            candidate_indices.extend(additional_indices)
    else:
        # Сортируем по доле совпадающих жанров
        partial_match_indices.sort(key=lambda x: x[1], reverse=True)
        candidate_indices = [i for i, _ in partial_match_indices[:min(top_n*3, len(partial_match_indices))]]

    # Этап 2: Ранжируем отобранные сериалы по дополнительным признакам

    # Вычисляем косинусное сходство для описаний
    target_desc_embedding = description_embeddings[idx].reshape(1, -1)

    # Вычисляем косинусное сходство для ключевых слов, если они доступны
    if keywords_embeddings is not None:
        target_keywords_embedding = keywords_embeddings[idx].reshape(1, -1)

    # Подготавливаем массив для хранения итоговых оценок
    final_scores = []

    for i in candidate_indices:
        # Проверяем, что индекс находится в пределах массива эмбеддингов
        if i >= len(description_embeddings):
            continue

        # Базовая оценка - сходство описаний
        desc_similarity = cosine_similarity(target_desc_embedding, description_embeddings[i].reshape(1, -1))[0][0]

        # Добавляем сходство по ключевым словам, если они доступны
        keywords_similarity = 0
        if keywords_embeddings is not None and i < len(keywords_embeddings):
            keywords_similarity = cosine_similarity(target_keywords_embedding, keywords_embeddings[i].reshape(1, -1))[0][0]

        # Комбинируем сходство описаний и ключевых слов
        combined_text_similarity = (1 - keywords_weight) * desc_similarity + keywords_weight * keywords_similarity

        # Бонусы за числовые признаки
        year_bonus = 0

        # Используем нормализованный год выпуска, если он доступен
        if 'release_year_norm' in data.columns:
            target_year = data.loc[idx, 'release_year_norm']
            show_year = data.loc[i, 'release_year_norm']
            if pd.notna(target_year) and pd.notna(show_year):
                year_diff = abs(target_year - show_year)
                year_bonus = max(0, 0.2 - year_diff * 0.2)  # Максимум 0.2 за совпадение года

        # Бонус за дополнительное совпадение стран производства
        country_bonus = 0
        if 'production_countries' in data.columns:
            # Используем напрямую списки стран, так как они уже в нужном формате
            target_countries = set([country.lower() for country in data.loc[idx, 'production_countries']]
                                  if isinstance(data.loc[idx, 'production_countries'], list) else [])
            show_countries = set([country.lower() for country in data.loc[i, 'production_countries']]
                                if isinstance(data.loc[i, 'production_countries'], list) else [])

            if target_countries and show_countries:
                country_intersection = len(target_countries.intersection(show_countries))
                country_union = len(target_countries.union(show_countries))
                country_bonus = 0.4 * (country_intersection / country_union)  # Максимум 0.4 за совпадение стран

        # Бонус за совпадение ключевых слов (прямое сравнение списков)
        keywords_bonus = 0
        if 'keywords' in data.columns:
            target_keywords = set(data.loc[idx, 'keywords'])
            show_keywords = set(data.loc[i, 'keywords'])
            if target_keywords and show_keywords:
                keywords_intersection = len(target_keywords.intersection(show_keywords))
                keywords_bonus = 0.3 * (keywords_intersection / len(target_keywords))  # Максимум 0.3 за совпадение ключевых слов

        # Итоговая оценка
        final_score = combined_text_similarity + country_bonus + year_bonus + keywords_bonus

        # Для сериалов с полным совпадением жанров добавляем большой бонус
        if i in exact_match_indices:
            final_score += 1.0  # Гарантируем, что сериалы с полным совпадением будут выше

        final_scores.append((i, final_score))

    # Сортируем по итоговой оценке
    final_scores.sort(key=lambda x: x[1], reverse=True)

    # Берем top_n лучших
    top_indices = [i for i, _ in final_scores[:min(top_n, len(final_scores))]]

    # Формируем результат
    columns_to_return = ['title', 'genres']
    for col in ['description', 'release_year', 'imdb_score']:
        if col in data.columns:
            columns_to_return.append(col)

    result_df = data.loc[top_indices, columns_to_return].copy()
    result_df['similarity_score'] = [score for _, score in final_scores[:len(top_indices)]]

    result_df = result_df.sort_values('similarity_score', ascending=False)
    rec_df = result_df.drop(columns=['similarity_score'])

    recommendations = rec_df.to_dict(orient='records')

    # Формируем финальный JSON
    result = {
        "recommendations": recommendations,
    }

    with open('recommendations.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# Пример использования:
if __name__ == "__main__":
    # Пути к файлам с предварительно обработанными данными
    data_path = "data_final.pkl"
    desc_embeddings_path = "description_embeddings.npy"
    keywords_embeddings_path = "keywords_embeddings.npy"

    # Загружаем данные
    data, description_embeddings, keywords_embeddings = load_recommendation_model(
        data_path, desc_embeddings_path, keywords_embeddings_path
    )

    # Запрашиваем у пользователя название сериала
    # show_title = input("Введите название сериала для поиска похожих: ")
    show_title = json_data['title']

    # Получаем рекомендации
    recommendations = get_recommendations(
        data, description_embeddings, keywords_embeddings,
        show_title, top_n=5
    )

    # Выводим результаты
    # if isinstance(recommendations, str):
    #     print(recommendations)  # Выводим сообщение об ошибке
    # else:
    #     print("\nРекомендованные сериалы:")
    #     # Форматируем вывод для лучшей читаемости
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.width', 1000)
    #     pd.set_option('display.max_colwidth', 50)
    #     print(recommendations)