from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import os

from database_postgres import get_db
from models_postgres import Show
from schemas_extended import ShowResponse, StatusResponse
from poster_api import PosterAPI

# Инициализируем API для работы с постерами
poster_api = PosterAPI()

# Создаем роутер для работы с постерами
router = APIRouter(
    prefix="/posters",
    tags=["posters"],
    responses={404: {"description": "Not found"}},
)

# Настройки для хранения постеров
POSTER_STORAGE_PATH = os.getenv("POSTER_STORAGE_PATH", "./posters")
if not os.path.exists(POSTER_STORAGE_PATH):
    os.makedirs(POSTER_STORAGE_PATH)

@router.post("/fetch-by-title", response_model=StatusResponse)
def fetch_poster_by_title(title: str, db: Session = Depends(get_db)):
    """
    Загрузка постера по названию сериала через TMDB API
    """
    show = db.query(Show).filter(Show.title == title).first()
    if not show:
        raise HTTPException(status_code=404, detail=f"Show with title '{title}' not found")
    
    # Получаем URL постера через API
    poster_url = poster_api.get_poster_url(title)
    if not poster_url:
        return {"status": "error", "message": "Poster not found"}
    
    # Обновляем URL постера в базе данных
    show.poster_url = poster_url
    
    # Сохраняем постер локально
    poster_path = poster_api.save_poster(title, show.id)
    if poster_path:
        show.poster_path = poster_path
    
    db.commit()
    return {"status": "ok", "message": "Poster fetched successfully"}

@router.post("/upload/{show_id}", response_model=StatusResponse)
def upload_poster(show_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Загрузка постера для сериала вручную
    """
    show = db.query(Show).filter(Show.id == show_id).first()
    if not show:
        raise HTTPException(status_code=404, detail=f"Show with ID {show_id} not found")
    
    # Проверяем расширение файла
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, PNG and WebP are allowed")
    
    # Сохраняем файл
    file_path = os.path.join(POSTER_STORAGE_PATH, f"show_{show_id}{file_extension}")
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    # Обновляем путь к постеру в базе данных
    show.poster_path = file_path
    db.commit()
    
    return {"status": "ok", "message": "Poster uploaded successfully"}

@router.get("/fetch-for-all", response_model=StatusResponse)
def fetch_posters_for_all_shows(db: Session = Depends(get_db)):
    """
    Загрузка постеров для всех сериалов в базе данных
    """
    shows = db.query(Show).all()
    success_count = 0
    failed_count = 0
    
    for show in shows:
        if not show.poster_url:
            # Получаем URL постера через API
            poster_url = poster_api.get_poster_url(show.title)
            if poster_url:
                show.poster_url = poster_url
                
                # Сохраняем постер локально
                poster_path = poster_api.save_poster(show.title, show.id)
                if poster_path:
                    show.poster_path = poster_path
                    success_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
    
    db.commit()
    return {
        "status": "ok", 
        "message": f"Posters fetched: {success_count} successful, {failed_count} failed"
    }

@router.get("/info/{show_id}", response_model=ShowResponse)
def get_poster_info(show_id: int, db: Session = Depends(get_db)):
    """
    Получение информации о постере сериала
    """
    show = db.query(Show).filter(Show.id == show_id).first()
    if not show:
        raise HTTPException(status_code=404, detail=f"Show with ID {show_id} not found")
    
    return show

@router.post("/refresh/{show_id}", response_model=StatusResponse)
def refresh_poster(show_id: int, db: Session = Depends(get_db)):
    """
    Обновление постера для сериала
    """
    show = db.query(Show).filter(Show.id == show_id).first()
    if not show:
        raise HTTPException(status_code=404, detail=f"Show with ID {show_id} not found")
    
    # Получаем URL постера через API
    poster_url = poster_api.get_poster_url(show.title)
    if not poster_url:
        return {"status": "error", "message": "Poster not found"}
    
    # Обновляем URL постера в базе данных
    show.poster_url = poster_url
    
    # Сохраняем постер локально
    poster_path = poster_api.save_poster(show.title, show.id)
    if poster_path:
        show.poster_path = poster_path
    
    db.commit()
    return {"status": "ok", "message": "Poster refreshed successfully"}
