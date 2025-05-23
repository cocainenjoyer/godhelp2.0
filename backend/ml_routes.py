from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json

from database_postgres import get_db
from models_postgres import Show, User, InitialPreferences, Recommendation, UserHistory
from schemas_extended import Movie, QuizRequest
from ml_integration import MLIntegration

# Инициализируем интеграцию с ML-моделями
ml_integration = MLIntegration()

# Создаем роутер для работы с рекомендациями
router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
    responses={404: {"description": "Not found"}},
)

@router.get("/profile", response_model=List[Movie])
def get_recommendations_by_profile(id: int = Query(...), db: Session = Depends(get_db)):
    """
    Получение рекомендаций на основе профиля пользователя
    """
    # Проверяем, существует ли пользователь
    user = db.query(User).filter(User.id == id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {id} not found")
    
    # Проверяем, есть ли у пользователя начальные предпочтения
    prefs = db.query(InitialPreferences).filter(InitialPreferences.user_id == id).first()
    if not prefs:
        raise HTTPException(status_code=404, detail="Initial profile not found for user")
    
    # Получаем существующие рекомендации из базы данных
    existing_recommendations = db.query(Recommendation).filter(
        Recommendation.user_id == id,
        Recommendation.recommendation_type == "profile"
    ).join(Show).all()
    
    # Если рекомендации уже есть, возвращаем их
    if existing_recommendations:
        return [
            {
                "id": rec.show.id,
                "title": rec.show.title,
                "description": rec.show.description or "Описание отсутствует",
                "image": rec.show.poster_url or "https://via.placeholder.com/200",
                "rating": rec.show.imdb_score or 0.0,
                "year": rec.show.release_year or 0,
                "genres": rec.show.genres
            }
            for rec in existing_recommendations
        ]
    
    # Если рекомендаций нет, генерируем новые с помощью ML-модели
    user_preferences = {
        "preferred_genres": prefs.genres,
        "release_year_range": prefs.period,
        "runtime_range": prefs.duration
    }
    
    recommendations = ml_integration.get_recommendations_by_profile(user_preferences)
    
    # Если модель не вернула рекомендации, используем заглушку
    if not recommendations:
        recommendations = [
            {
                "id": 1,
                "title": "Fake Series 1",
                "description": "Описание 1",
                "image": "https://via.placeholder.com/200",
                "rating": 7.5,
                "year": 2021,
                "genres": prefs.genres
            },
            {
                "id": 2,
                "title": "Fake Series 2",
                "description": "Описание 2",
                "image": "https://via.placeholder.com/200",
                "rating": 8.1,
                "year": 2022,
                "genres": prefs.genres
            }
        ]
    
    # Сохраняем рекомендации в базу данных
    for rec in recommendations:
        # Проверяем, существует ли сериал в базе
        show = db.query(Show).filter(Show.id == rec["id"]).first()
        if not show:
            # Если сериала нет, создаем его
            show = Show(
                id=rec["id"],
                title=rec["title"],
                description=rec["description"],
                release_year=rec["year"],
                genres=rec["genres"],
                imdb_score=rec["rating"],
                poster_url=rec["image"]
            )
            db.add(show)
            db.commit()
            db.refresh(show)
        
        # Создаем запись о рекомендации
        recommendation = Recommendation(
            user_id=id,
            show_id=show.id,
            recommendation_type="profile",
            score=rec["rating"]
        )
        db.add(recommendation)
    
    db.commit()
    
    return recommendations

@router.post("/by-quiz", response_model=List[Movie])
def recommend_by_quiz(request: QuizRequest, user_id: int = Query(None), db: Session = Depends(get_db)):
    """
    Получение рекомендаций на основе ответов на анкету
    """
    # Сохраняем ответы на анкету, если указан пользователь
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
    
    # Получаем рекомендации от ML-модели
    quiz_answers = [answer.dict() for answer in request.data]
    recommendations = ml_integration.get_recommendations_by_quiz(quiz_answers)
    
    # Если модель не вернула рекомендации, используем заглушку
    if not recommendations:
        recommendations = [
            {
                "id": 101,
                "title": "Quiz Based Drama",
                "description": "A show that matches your mood and vibe.",
                "image": "https://via.placeholder.com/200",
                "rating": 8.3,
                "year": 2020,
                "genres": ["drama", "romance"]
            },
            {
                "id": 102,
                "title": "Lighthearted Comedy",
                "description": "A perfect watch with friends!",
                "image": "https://via.placeholder.com/200",
                "rating": 7.9,
                "year": 2021,
                "genres": ["comedy", "family"]
            }
        ]
    
    # Сохраняем рекомендации в базу данных, если указан пользователь
    if user_id:
        for rec in recommendations:
            # Проверяем, существует ли сериал в базе
            show = db.query(Show).filter(Show.id == rec["id"]).first()
            if not show:
                # Если сериала нет, создаем его
                show = Show(
                    id=rec["id"],
                    title=rec["title"],
                    description=rec["description"],
                    release_year=rec["year"],
                    genres=rec["genres"],
                    imdb_score=rec["rating"],
                    poster_url=rec["image"]
                )
                db.add(show)
                db.commit()
                db.refresh(show)
            
            # Создаем запись о рекомендации
            recommendation = Recommendation(
                user_id=user_id,
                show_id=show.id,
                recommendation_type="quiz",
                score=rec["rating"]
            )
            db.add(recommendation)
            
            # Добавляем запись в историю
            history = UserHistory(
                user_id=user_id,
                show_id=show.id,
                interaction_type="recommend_by_quiz"
            )
            db.add(history)
        
        db.commit()
    
    return recommendations

@router.get("/similar-by-id", response_model=List[Movie])
def similar_by_id(id: int = Query(...), user_id: int = Query(None), db: Session = Depends(get_db)):
    """
    Получение похожих сериалов по ID
    """
    # Проверяем, существует ли сериал
    show = db.query(Show).filter(Show.id == id).first()
    if not show:
        raise HTTPException(status_code=404, detail=f"Show with ID {id} not found")
    
    # Получаем похожие сериалы от ML-модели
    similar_shows = ml_integration.get_similar_shows(show.title)
    
    # Если модель не вернула рекомендации, используем заглушку
    if not similar_shows:
        similar_shows = [
            {
                "id": 301,
                "title": "Похожий сериал 1",
                "description": f"На основе сериала #{id}",
                "image": "https://via.placeholder.com/200",
                "rating": 8.1,
                "year": 2020,
                "genres": ["drama", "sci-fi"]
            },
            {
                "id": 302,
                "title": "Похожий сериал 2",
                "description": f"Альтернатива для сериала #{id}",
                "image": "https://via.placeholder.com/200",
                "rating": 7.9,
                "year": 2018,
                "genres": ["mystery", "thriller"]
            }
        ]
    
    # Сохраняем рекомендации в базу данных, если указан пользователь
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
            
        for rec in similar_shows:
            # Проверяем, существует ли сериал в базе
            similar_show = db.query(Show).filter(Show.id == rec["id"]).first()
            if not similar_show:
                # Если сериала нет, создаем его
                similar_show = Show(
                    id=rec["id"],
                    title=rec["title"],
                    description=rec["description"],
                    release_year=rec["year"],
                    genres=rec["genres"],
                    imdb_score=rec["rating"],
                    poster_url=rec["image"]
                )
                db.add(similar_show)
                db.commit()
                db.refresh(similar_show)
            
            # Создаем запись о рекомендации
            recommendation = Recommendation(
                user_id=user_id,
                show_id=similar_show.id,
                recommendation_type="similar",
                source_show_id=id,
                score=rec["rating"]
            )
            db.add(recommendation)
            
            # Добавляем запись в историю
            history = UserHistory(
                user_id=user_id,
                show_id=similar_show.id,
                interaction_type="similar"
            )
            db.add(history)
        
        db.commit()
    
    return similar_shows

@router.post("/import-shows", response_model=Dict[str, Any])
def import_shows_from_dataset(limit: int = Query(100), db: Session = Depends(get_db)):
    """
    Импорт сериалов из датасета в базу данных
    """
    shows = ml_integration.import_shows_from_dataset(limit)
    
    if not shows:
        return {"status": "error", "message": "Failed to import shows", "count": 0}
    
    imported_count = 0
    for show_data in shows:
        # Проверяем, существует ли сериал в базе
        show = db.query(Show).filter(Show.title == show_data["title"]).first()
        if not show:
            # Если сериала нет, создаем его
            show = Show(
                title=show_data["title"],
                description=show_data["description"],
                release_year=show_data["release_year"],
                genres=show_data["genres"],
                runtime=show_data["runtime"],
                imdb_score=show_data["imdb_score"],
                tmdb_score=show_data["tmdb_score"],
                tmdb_popularity=show_data["tmdb_popularity"],
                age_certification=show_data["age_certification"],
                production_countries=show_data["production_countries"],
                keywords=show_data["keywords"]
            )
            db.add(show)
            imported_count += 1
    
    db.commit()
    
    return {
        "status": "ok", 
        "message": f"Successfully imported {imported_count} shows", 
        "count": imported_count
    }
