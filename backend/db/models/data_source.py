from sqlalchemy import Column, Integer, String, Enum, ForeignKey
from sqlalchemy.orm import relationship
from backend.db.models.base import Base
import enum

class DataSource(Base):
    __tablename__ = 'data_sources'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True)
    db_type = Column(String, default="postgresql", nullable=False)  # PostgreSQL, MySQL
    connection_string = Column(String)  # This can be the connection URL or credentials
    # Relationships for databases, tables, and columns
    databases = relationship("Database", back_populates="data_source")

class Database(Base):
    __tablename__ = 'databases'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    data_source_id = Column(Integer, ForeignKey('data_sources.id'))
    data_source = relationship("DataSource", back_populates="databases")
    tables = relationship("Table", back_populates="database")

class Table(Base):
    __tablename__ = 'tables'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    database_id = Column(Integer, ForeignKey('databases.id'))

    database = relationship("Database", back_populates="tables")
    columns = relationship("Column", back_populates="table")

class Column(Base):
    __tablename__ = 'columns'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    data_type = Column(String)
    table_id = Column(Integer, ForeignKey('tables.id'))

    table = relationship("Table", back_populates="columns")
