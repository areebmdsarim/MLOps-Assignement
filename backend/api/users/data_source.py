from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.db.models.data_source import DataSource, Database, Table, Column
from backend.db.models.base import SessionLocal
import psycopg2  # PostgreSQL
import pymysql  # MySQL

router = APIRouter(
    prefix='/data_sources',
    tags=['data_sources']
)

class CreateDataSourceRequest(BaseModel):
    name: str
    db_type: str  # 'postgresql' or 'mysql'
    connection_string: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_data_source(db: db_dependency, create_data_source_request: CreateDataSourceRequest):
    # Validate the db type
    if create_data_source_request.db_type not in ['postgresql', 'mysql']:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid database type")
    
    # Create data source entry in the database
    data_source = DataSource(
        name=create_data_source_request.name,
        db_type=create_data_source_request.db_type,
        connection_string=create_data_source_request.connection_string
    )

    db.add(data_source)
    db.commit()
    db.refresh(data_source)

    # Fetch and store database, tables, and columns
    if create_data_source_request.db_type == 'postgresql':
        # Fetch data from PostgreSQL
        fetch_postgresql_data(data_source, db)
    elif create_data_source_request.db_type == 'mysql':
        # Fetch data from MySQL
        fetch_mysql_data(data_source, db)
    
    return {"message": "Data source created successfully"}

def fetch_postgresql_data(data_source, db):
    try:
        # Connect to PostgreSQL and fetch databases, tables, and columns
        conn = psycopg2.connect(data_source.connection_string)
        cursor = conn.cursor()
        
        # Fetch databases
        cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
        databases = cursor.fetchall()
        for db_name in databases:
            db_entry = Database(name=db_name[0], data_source_id=data_source.id)
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)
            
            # Fetch tables for each database
            cursor.execute(f"SELECT table_name FROM information_schema.tables WHERE table_catalog='{db_name[0]}';")
            tables = cursor.fetchall()
            for table_name in tables:
                table_entry = Table(name=table_name[0], database_id=db_entry.id)
                db.add(table_entry)
                db.commit()
                db.refresh(table_entry)
                
                # Fetch columns for each table
                cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table_name[0]}';")
                columns = cursor.fetchall()
                for column_name, data_type in columns:
                    column_entry = Column(name=column_name, data_type=data_type, table_id=table_entry.id)
                    db.add(column_entry)
                    db.commit()
        
        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching PostgreSQL data: {str(e)}")

def fetch_mysql_data(data_source, db):
    try:
        # Connect to MySQL and fetch databases, tables, and columns
        conn = pymysql.connect(data_source.connection_string)
        cursor = conn.cursor()

        # Fetch databases
        cursor.execute("SHOW DATABASES;")
        databases = cursor.fetchall()
        for db_name in databases:
            db_entry = Database(name=db_name[0], data_source_id=data_source.id)
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)

            # Fetch tables for each database
            cursor.execute(f"USE {db_name[0]};")
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            for table_name in tables:
                table_entry = Table(name=table_name[0], database_id=db_entry.id)
                db.add(table_entry)
                db.commit()
                db.refresh(table_entry)

                # Fetch columns for each table
                cursor.execute(f"DESCRIBE {table_name[0]};")
                columns = cursor.fetchall()
                for column_name, *_ in columns:
                    column_entry = Column(name=column_name, data_type="Unknown", table_id=table_entry.id)  # MySQL doesn't return data type in a direct way, use `DESCRIBE`
                    db.add(column_entry)
                    db.commit()

        cursor.close()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching MySQL data: {str(e)}")
