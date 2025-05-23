from sqlalchemy import Column, Integer, String, ForeignKey, Float, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from database_postgres import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    preferences = relationship("InitialPreferences", back_populates="user", uselist=False)
    history = relationship("UserHistory", back_populates="user")
    quiz_answers = relationship("QuizAnswers", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")
    sessions = relationship("Session", back_populates="user")

class InitialPreferences(Base):
    __tablename__ = "initial_preferences"
    
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    genres = Column(JSONB, nullable=False)
    period = Column(String(50), nullable=False)
    duration = Column(String(50), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="preferences")

class Show(Base):
    __tablename__ = "shows"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(String, nullable=True)
    release_year = Column(Integer, nullable=True)
    genres = Column(JSONB, nullable=False)
    runtime = Column(Integer, nullable=True)
    imdb_score = Column(Float, nullable=True)
    tmdb_score = Column(Float, nullable=True)
    tmdb_popularity = Column(Float, nullable=True)
    age_certification = Column(String(10), nullable=True)
    production_countries = Column(JSONB, nullable=True)
    keywords = Column(JSONB, nullable=True)
    poster_url = Column(String(255), nullable=True)
    poster_path = Column(String(255), nullable=True)
    
    # Relationships
    history = relationship("UserHistory", back_populates="show")
    recommendations = relationship("Recommendation", 
                                  foreign_keys="Recommendation.show_id", 
                                  back_populates="show")
    source_recommendations = relationship("Recommendation", 
                                         foreign_keys="Recommendation.source_show_id", 
                                         back_populates="source_show")
    
    # Indexes
    __table_args__ = (
        Index('idx_shows_title', title),
        Index('idx_shows_release_year', release_year),
    )

class UserHistory(Base):
    __tablename__ = "user_history"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    show_id = Column(Integer, ForeignKey("shows.id", ondelete="CASCADE"))
    interaction_type = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="history")
    show = relationship("Show", back_populates="history")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'show_id', 'interaction_type', name='uq_user_show_interaction'),
        Index('idx_user_history_user_id', user_id),
    )

class QuizAnswers(Base):
    __tablename__ = "quiz_answers"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    quiz_data = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="quiz_answers")

class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    show_id = Column(Integer, ForeignKey("shows.id", ondelete="CASCADE"))
    recommendation_type = Column(String(50), nullable=False)
    source_show_id = Column(Integer, ForeignKey("shows.id"), nullable=True)
    score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="recommendations")
    show = relationship("Show", foreign_keys=[show_id], back_populates="recommendations")
    source_show = relationship("Show", foreign_keys=[source_show_id], back_populates="source_recommendations")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'show_id', 'recommendation_type', 'source_show_id', name='uq_user_show_rec_type_source'),
        Index('idx_recommendations_user_id', user_id),
        Index('idx_recommendations_type', recommendation_type),
    )

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    session_token = Column(String(255), unique=True, nullable=False)
    expires = Column(DateTime(timezone=True), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
