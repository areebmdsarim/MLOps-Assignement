from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL_POSTGRESQL = "postgresql://postgres:password@localhost:5432/MLOPS"
SQLALCHEMY_DATABASE_URL_MYSQL = "mysql+mysqlconnector://root:password@localhost:3306/MLOPS"

Postgresql_engine = create_engine(SQLALCHEMY_DATABASE_URL_POSTGRESQL)
Postgresql_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=Postgresql_engine)

MySQL_engine = create_engine(SQLALCHEMY_DATABASE_URL_MYSQL)
MySQL_SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=MySQL_engine)
Base = declarative_base()