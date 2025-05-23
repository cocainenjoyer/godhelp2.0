import os
from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import json
import bcrypt
import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

# Импортируем модули для работы с базой данных
from database_postgres import SessionLocal, engine, Base
from models_postgres import User, InitialPreferences, Show, UserHistory, QuizAnswers, Recommendation, Session as DbSession
from schemas_extended import (
    UserCreate, UserLogin, UserResponse, RegisterInitialRequest, RegisterInitialResponse,
    ShowCreate, ShowResponse, Movie, QuizRequest, ShowMapItem, TokenResponse, StatusResponse,
    HistoryResponse, RecommendationCreate, RecommendationResponse
)

# Импортируем роутеры
from poster_routes import router as poster_router
from ml_routes import router as ml_router

# Создаем таблицы в базе данных
Base.metadata.create_all(bind=engine)

# Инициализируем FastAPI приложение
app = FastAPI(title="Movie Recommender API")

# Настраиваем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене заменить на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройки для JWT
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Настройки для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Настройки для OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# Настройки для хранения постеров
POSTER_STORAGE_PATH = os.getenv("POSTER_STORAGE_PATH", "./posters")
if not os.path.exists(POSTER_STORAGE_PATH):
    os.makedirs(POSTER_STORAGE_PATH)

# Функция для получения сессии базы данных
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Вспомогательные функции для аутентификации
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

# Подключаем роутеры
app.include_router(poster_router)
app.include_router(ml_router)

# Монтируем статические файлы для постеров
app.mount("/posters", StaticFiles(directory=POSTER_STORAGE_PATH), name="posters")

# Эндпоинты для аутентификации
@app.post("/auth/register", response_model=TokenResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        username=user.username,
        password_hash=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(db_user.id)}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/auth/login", response_model=TokenResponse)
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    # Обновляем время последнего входа
    db_user.last_login = datetime.utcnow()
    db.commit()
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(db_user.id)}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Эндпоинты для работы с начальными предпочтениями
@app.post("/users/register-initial", response_model=RegisterInitialResponse)
def register_initial(request: RegisterInitialRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.id).first()
    if not user:
        user = User(
            id=request.id, 
            email=request.email, 
            username=request.username,
            password_hash=get_password_hash("temporary_password")  # Временный пароль
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    preferences = db.query(InitialPreferences).filter(InitialPreferences.user_id == user.id).first()
    if not preferences:
        preferences = InitialPreferences(
            user_id=user.id,
            genres=request.initial.genres,
            period=request.initial.period,
            duration=request.initial.duration
        )
        db.add(preferences)
    else:
        preferences.genres = request.initial.genres
        preferences.period = request.initial.period
        preferences.duration = request.initial.duration

    db.commit()
    return {"status": "ok"}

# Эндпоинты для работы с сериалами
@app.get("/titles/map", response_model=List[ShowMapItem])
def get_title_map(db: Session = Depends(get_db)):
    shows = db.query(Show).order_by(Show.title.asc()).all()
    return [{"id": show.id, "title": show.title} for show in shows]

@app.post("/titles/init-demo", response_model=StatusResponse)
def init_demo_titles(db: Session = Depends(get_db)):
    from poster_api import PosterAPI
    poster_api = PosterAPI()
    
    demo_titles = ["Breaking Bad", "The Office", "Game of Thrones", "Friends", "Sherlock"]
    for idx, title in enumerate(demo_titles):
        if not db.query(Show).filter_by(title=title).first():
            # Получаем постер через API
            poster_url = poster_api.get_poster_url(title)
            
            show = Show(
                id=idx + 1, 
                title=title,
                genres=["drama"],  # Заглушка
                poster_url=poster_url
            )
            db.add(show)
            
            # Если получили URL постера, сохраняем его локально
            if poster_url:
                poster_path = poster_api.save_poster(title, idx + 1)
                if poster_path:
                    show.poster_path = poster_path
    
    db.commit()
    return {"status": "ok", "message": "Demo titles initialized"}

# Эндпоинты для работы с историей пользователя
@app.get("/users/{user_id}/history", response_model=HistoryResponse)
def get_user_history(user_id: int, db: Session = Depends(get_db)):
    history = db.query(UserHistory).filter(UserHistory.user_id == user_id).all()
    return {"history": history}

@app.post("/users/{user_id}/history", response_model=StatusResponse)
def add_to_history(user_id: int, show_id: int = Query(...), interaction_type: str = Query(...), db: Session = Depends(get_db)):
    # Проверяем, существует ли пользователь
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found")
    
    # Проверяем, существует ли сериал
    show = db.query(Show).filter(Show.id == show_id).first()
    if not show:
        raise HTTPException(status_code=404, detail=f"Show with ID {show_id} not found")
    
    # Проверяем, есть ли уже такая запись в истории
    existing_history = db.query(UserHistory).filter(
        UserHistory.user_id == user_id,
        UserHistory.show_id == show_id,
        UserHistory.interaction_type == interaction_type
    ).first()
    
    if existing_history:
        # Обновляем время создания
        existing_history.created_at = datetime.utcnow()
    else:
        # Создаем новую запись
        history = UserHistory(
            user_id=user_id,
            show_id=show_id,
            interaction_type=interaction_type
        )
        db.add(history)
    
    db.commit()
    return {"status": "ok", "message": "History updated successfully"}

# Отладочные эндпоинты
@app.get("/debug/db")
def debug_database(db: Session = Depends(get_db)):
    users = db.query(User).all()
    prefs = db.query(InitialPreferences).all()
    shows = db.query(Show).all()
    history = db.query(UserHistory).all()
    recommendations = db.query(Recommendation).all()

    return {
        "users": [
            {"id": u.id, "email": u.email, "username": u.username}
            for u in users
        ],
        "initial_preferences": [
            {
                "user_id": p.user_id,
                "genres": p.genres,
                "period": p.period,
                "duration": p.duration
            }
            for p in prefs
        ],
        "shows": [
            {"id": s.id, "title": s.title, "poster_url": s.poster_url} for s in shows
        ],
        "history": [
            {"user_id": h.user_id, "show_id": h.show_id, "interaction_type": h.interaction_type}
            for h in history
        ],
        "recommendations": [
            {"user_id": r.user_id, "show_id": r.show_id, "type": r.recommendation_type}
            for r in recommendations
        ]
    }

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
