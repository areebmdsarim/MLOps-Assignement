from typing import Annotated, List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.db.models.data_source import DataSource, Database, Table, Column
from backend.db.models.base import Postgresql_SessionLocal
from backend.api.users.auth import Postgresql_db_dependency, MySQL_db_dependency
from urllib.parse import urlparse
from dotenv import load_dotenv
import requests
import pymysql
import google.generativeai as genai
import os
import pandas as pd
from backend.db.utils import infer_relationships
from fuzzywuzzy import fuzz
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer




router = APIRouter(
    prefix="/data_sources",
    tags=["data_sources"],
)

# qa_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Pydantic Request Models
class CreateDataSourceRequest(BaseModel):
    name: str
    db_type: str  # 'postgresql'
    connection_string: str

class DatabaseResponse(BaseModel):
    name: str

class TableResponse(BaseModel):
    name: str

class ColumnResponse(BaseModel):
    name: str
    data_type: str

class TableDependencyResponse(BaseModel):
    table_name: str
    related_table: str
    related_column: str
    relationship_type: str  # "foreign_key", "heuristic", etc.
    confidence_score: float  # Confidence score for ML models


class DataSourceResponse(BaseModel):
    id: int
    name: str
    db_type: str
    connection_string: str
    databases: List[DatabaseResponse]

class TableDescriptionResponse(BaseModel):
    table_description: str
    columns: List[Dict[str, str]]  # [{"name": "column_name", "description": "description"}]


# class QARequest(BaseModel):
#     question: str
#     data_source_id: int

# class QAResponse(BaseModel):
#     answer: str

    class Config:
        orm_mode = True


@router.post("/PostgreSQL", response_model=DataSourceResponse, status_code=status.HTTP_201_CREATED)
async def create__postgresql_data_source(
    create_data_source_request: CreateDataSourceRequest,
    db: Postgresql_db_dependency,
):
    # Validate DB type
    if create_data_source_request.db_type != "postgresql":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only PostgreSQL is supported."
        )

    # Check if data source exists
    if db.query(DataSource).filter(DataSource.name == create_data_source_request.name).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Data source already exists.")

    # Create and save data source
    data_source = DataSource(
        name=create_data_source_request.name,
        db_type=create_data_source_request.db_type,
        connection_string=create_data_source_request.connection_string,
    )
    db.add(data_source)
    db.commit()
    db.refresh(data_source)

    # Fetch PostgreSQL metadata
    try:
        fetch_postgresql_data(data_source, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch PostgreSQL metadata: {str(e)}")

    return data_source


def fetch_postgresql_data(data_source: DataSource, db: Session):
    import psycopg2

    conn = psycopg2.connect(data_source.connection_string)
    cursor = conn.cursor()

    # Fetch databases
    cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
    for db_name, in cursor.fetchall():
        # Check if database already exists
        db_entry = (
            db.query(Database)
            .filter(Database.name == db_name, Database.data_source_id == data_source.id)
            .first()
        )
        if not db_entry:
            db_entry = Database(name=db_name, data_source_id=data_source.id)
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)

        # Fetch tables
        cursor.execute(
            f"""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_catalog = '{db_name}';
            """
        )
        for table_name, in cursor.fetchall():
            # Check if table already exists
            table_entry = (
                db.query(Table)
                .filter(Table.name == table_name, Table.database_id == db_entry.id)
                .first()
            )
            if not table_entry:
                table_entry = Table(name=table_name, database_id=db_entry.id)
                db.add(table_entry)
                db.commit()
                db.refresh(table_entry)

            # Fetch columns
            cursor.execute(
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = '{table_name}';
                """
            )
            for column_name, data_type in cursor.fetchall():
                # Check if column already exists
                column_entry = (
                    db.query(Column)
                    .filter(
                        Column.name == column_name, Column.table_id == table_entry.id
                    )
                    .first()
                )
                if not column_entry:
                    column_entry = Column(
                        name=column_name, data_type=data_type, table_id=table_entry.id
                    )
                    db.add(column_entry)
                    db.commit()

    cursor.close()
    conn.close()

@router.post("/MYSQL", response_model=DataSourceResponse, status_code=status.HTTP_201_CREATED)
async def create_MySQL_data_source(
    create_data_source_request: CreateDataSourceRequest,
    db: MySQL_db_dependency,
):
    # Validate DB type
    if create_data_source_request.db_type != "MySQL":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Only MySQL is supported."
        )

    # Check if data source exists
    if db.query(DataSource).filter(DataSource.name == create_data_source_request.name).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Data source already exists.")

    # Create and save data source
    data_source = DataSource(
        name=create_data_source_request.name,
        db_type=create_data_source_request.db_type,
        connection_string=create_data_source_request.connection_string,
    )
    db.add(data_source)
    db.commit()
    db.refresh(data_source)

    # Fetch PostgreSQL metadata
    try:
        fetch_MySQL_data(data_source, db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch MySQL metadata: {str(e)}")

    return data_source

def parse_MySQL_connection_string(connection_string):
    parsed = urlparse(connection_string)
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": parsed.username,
        "password": parsed.password,
        "database": parsed.path.lstrip("/")  # Removes leading '/'
    }

def fetch_MySQL_data(data_source: DataSource, db: Session):
    connection_params = parse_MySQL_connection_string(data_source.connection_string)

    # Connect to MySQL
    conn = pymysql.connect(
        host=connection_params["host"],
        port=connection_params["port"],
        user=connection_params["user"],
        password=connection_params["password"],
        database=connection_params["database"]
    )
    cursor = conn.cursor()

    # Fetch databases
    cursor.execute("SHOW DATABASES;")
    for db_name, in cursor.fetchall():
        # Check if database already exists
        db_entry = (
            db.query(Database)
            .filter(Database.name == db_name, Database.data_source_id == data_source.id)
            .first()
        )
        if not db_entry:
            db_entry = Database(name=db_name, data_source_id=data_source.id)
            db.add(db_entry)
            db.commit()
            db.refresh(db_entry)

        # Fetch tables
        cursor.execute(f"SHOW TABLES FROM `{db_name}`;")
        for table_name, in cursor.fetchall():
            # Check if table already exists
            table_entry = (
                db.query(Table)
                .filter(Table.name == table_name, Table.database_id == db_entry.id)
                .first()
            )
            if not table_entry:
                table_entry = Table(name=table_name, database_id=db_entry.id)
                db.add(table_entry)
                db.commit()
                db.refresh(table_entry)

            # Fetch columns
            cursor.execute(
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = '{db_name}' AND table_name = '{table_name}';
                """
            )
            for column_name, data_type in cursor.fetchall():
                # Check if column already exists
                column_entry = (
                    db.query(Column)
                    .filter(
                        Column.name == column_name, Column.table_id == table_entry.id
                    )
                    .first()
                )
                if not column_entry:
                    column_entry = Column(
                        name=column_name, data_type=data_type, table_id=table_entry.id
                    )
                    db.add(column_entry)
                    db.commit()

    cursor.close()
    conn.close()


def get_gemini_description(prompt: str) -> str:
    """
    Interacts with the Gemini API to get a description for a given prompt.
    """
    
    load_dotenv()

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    model=genai.GenerativeModel("gemini-1.5-flash-latest")
    try:
        # Use the model to generate text based on the prompt
        response = model.generate_content(
            contents=[{"parts": [{"text": prompt}]}]
        )

        # Extract and return the text content from the response
        return response.candidates[0].content.parts[0].text

    except Exception as e:
        # If there's any error, raise a 500 internal server error with a message
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate description using Gemini API: {str(e)}",
        )

@router.get(
    "/{data_source_id}/databases/{database_name}/tables/{table_name}/describe",
    response_model=TableDescriptionResponse,
)
async def describe_table(
    data_source_id: int,
    database_name: str,
    table_name: str,
    db: Postgresql_db_dependency,
):
    # Fetch the data source
    data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if not data_source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Data source not found."
        )

    # Connect to the database
    import psycopg2

    try:
        conn = psycopg2.connect(data_source.connection_string)
        cursor = conn.cursor()

        # Describe the table structure
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = 'public';")
        columns = cursor.fetchall()
        if not columns:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Table not found."
            )

        # Generate descriptions using Gemini API
        table_prompt = f"Describe the purpose and structure of the table '{table_name}'."
        table_description = get_gemini_description(table_prompt)
        
        column_descriptions = []
        for column_name, data_type in columns:
            column_prompt = (
                f"Describe the column '{column_name}' with data type '{data_type}' in the table '{table_name}'."
            )
            column_description = get_gemini_description(column_prompt)
            column_descriptions.append(
                {"name": column_name, "description": column_description}
            )

        return TableDescriptionResponse(
            table_description=table_description, columns=column_descriptions
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to describe table: {str(e)}",
        )
    finally:
        if conn:
            cursor.close()
            conn.close()

@router.get(
    "/{data_source_id}/table_dependencies",
    response_model=List[TableDependencyResponse],
)
async def find_table_dependencies(data_source_id: int, db: Postgresql_db_dependency):
    """
    Find tables with column dependencies using a machine learning model or heuristics.
    """
    # Fetch all tables and columns for the data source
    tables = (
        db.query(Table)
        .filter(Table.database_id.in_(
            db.query(Database.id).filter(Database.data_source_id == data_source_id)
        ))
        .all()
    )

    # Create a dictionary to store tables and their columns
    table_columns = {}
    for table in tables:
        table_columns[table.name] = [
            {"name": column.name, "data_type": column.data_type}
            for column in table.columns
        ]

    dependencies = []

    # Compare columns between tables
    for table_name, columns in table_columns.items():
        for other_table_name, other_columns in table_columns.items():
            if table_name == other_table_name:
                continue  # Skip comparing the same table

            for column in columns:
                for other_column in other_columns:
                    if column["data_type"] == other_column["data_type"]:
                        # Use ML to infer the relationship between columns
                        data1 = pd.DataFrame({column["name"]: [value for value in range(100)]})  # Sample data for column 1
                        data2 = pd.DataFrame({other_column["name"]: [value for value in range(100)]})
                         # Sample data for column 2

                        # Get ML-based relationships and confidence scores
                        inferred_relationships = infer_relationships(data1, data2)
                        if infer_relationships :
                        # If the confidence score is above a threshold, add it to the dependencies
                            for relationship in inferred_relationships:
                                if relationship['confidence_score'] > 0.7:  # Example threshold
                                    dependencies.append(
                                        TableDependencyResponse(
                                            table_name=table_name,
                                            related_table=other_table_name,
                                            related_column=column["name"] + " " + other_column["name"],
                                            relationship_type="ml_inferred",  # Using ML inference
                                            confidence_score=relationship['confidence_score'],
                                        )
                                    )

    if not dependencies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No table dependencies found.",
        )

    return dependencies


# @router.post("/qna", response_model=QAResponse)
# async def ask_question(
#     qa_request: QARequest,
#     db: db_dependency,
# ):
#     """
#     Endpoint to handle Q&A for the data models.
#     """
#     # Fetch data source
#     data_source = db.query(DataSource).filter(DataSource.id == qa_request.data_source_id).first()
#     if not data_source:
#         raise HTTPException(status_code=404, detail="Data source not found")

#     # Fetch databases, tables, and columns
#     databases = data_source.databases
#     metadata = []
#     for database in databases:
#         for table in database.tables:
#             table_metadata = {
#                 "table_name": table.name,
#                 "columns": [{"name": col.name, "data_type": col.data_type} for col in table.columns]
#             }
#             metadata.append(table_metadata)

#     # Generate a knowledge base string
#     knowledge_base = []
#     for item in metadata:
#         columns_info = ", ".join([f"{col['name']} ({col['data_type']})" for col in item['columns']])
#         knowledge_base.append(f"Table {item['table_name']} has columns: {columns_info}")
#     knowledge_base_text = " ".join(knowledge_base)

#     # Prepare embeddings for the knowledge base and the question
#     knowledge_embeddings = qa_model.encode([knowledge_base_text])
#     question_embedding = qa_model.encode([qa_request.question])

#     # Calculate similarity
#     similarity_score = cosine_similarity(knowledge_embeddings, question_embedding)[0][0]

#     # Generate a response
#     if similarity_score > 0:  # Example threshold for matching
#         answer = f"Based on the knowledge base: {knowledge_embeddings}"
#     else:
#         answer = "I'm sorry, I couldn't find a relevant answer to your question."

#     return QAResponse(answer=answer)