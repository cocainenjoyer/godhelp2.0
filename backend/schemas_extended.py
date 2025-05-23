from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal, Optional, Dict, Any, Union
from datetime import datetime

# Базовые схемы для аутентификации
class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# Схемы для начальных предпочтений
class InitialProfile(BaseModel):
    genres: List[str]
    period: str
    duration: str

class RegisterInitialRequest(BaseModel):
    id: int
    email: EmailStr
    username: str
    initial: InitialProfile

class RegisterInitialResponse(BaseModel):
    status: str

# Схемы для сериалов
class ShowBase(BaseModel):
    title: str
    description: Optional[str] = None
    release_year: Optional[int] = None
    genres: List[str]
    runtime: Optional[int] = None
    imdb_score: Optional[float] = None
    tmdb_score: Optional[float] = None
    tmdb_popularity: Optional[float] = None
    age_certification: Optional[str] = None
    production_countries: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

class ShowCreate(ShowBase):
    pass

class ShowUpdate(ShowBase):
    pass

class ShowResponse(ShowBase):
    id: int
    poster_url: Optional[str] = None

    class Config:
        orm_mode = True

class Movie(BaseModel):
    id: int
    title: str
    description: str
    image: str
    rating: float
    year: int
    genres: List[str]

# Схемы для анкеты
class QuizAnswer(BaseModel):
    id: Literal["mood", "country", "company"]
    option: str

class QuizRequest(BaseModel):
    data: List[QuizAnswer]

class QuizResponse(BaseModel):
    id: int
    recommendations: List[Movie]

# Схемы для истории пользователя
class HistoryEntry(BaseModel):
    show_id: int
    interaction_type: str
    created_at: datetime

    class Config:
        orm_mode = True

class HistoryResponse(BaseModel):
    history: List[HistoryEntry]

# Схемы для рекомендаций
class RecommendationBase(BaseModel):
    show_id: int
    recommendation_type: str
    source_show_id: Optional[int] = None
    score: Optional[float] = None

class RecommendationCreate(RecommendationBase):
    user_id: int

class RecommendationResponse(RecommendationBase):
    id: int
    created_at: datetime
    show: ShowResponse

    class Config:
        orm_mode = True

# Вспомогательные схемы
class ShowMapItem(BaseModel):
    id: int
    title: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
