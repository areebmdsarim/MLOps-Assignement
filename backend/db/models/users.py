from backend.db.models.base import Base #database
from sqlalchemy import Column, Integer, String, Boolean

class Users(Base):
    __tablename__ = 'users'

    username  = Column(String, primary_key=True, unique= True)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)