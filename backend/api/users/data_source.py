from typing import Annotated, List, Dict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.db.models.data_source import DataSource, Database, Table, Column
from backend.db.models.base import SessionLocal
from backend.api.users.auth import db_dependency
from dotenv import load_dotenv
import requests
import google.generativeai as genai
import os


router = APIRouter(
    prefix="/data_sources",
    tags=["data_sources"],
)

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

class DataSourceResponse(BaseModel):
    id: int
    name: str
    db_type: str
    connection_string: str
    databases: List[DatabaseResponse]

    class Config:
        orm_mode = True


@router.post("/", response_model=DataSourceResponse, status_code=status.HTTP_201_CREATED)
async def create_data_source(
    create_data_source_request: CreateDataSourceRequest,
    db: db_dependency,
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

class TableDescriptionResponse(BaseModel):
    table_description: str
    columns: List[Dict[str, str]]  # [{"name": "column_name", "description": "description"}]


def get_gemini_description(prompt: str) -> str:
    """
    Interacts with the Gemini API to get a description for a given prompt.
    """
    # api_url = "https://aistudio.google.com/app/apikey"
    # api_key = "AIzaSyDCDCXIXQ416cKAWaFzDkTvEVfCqexI0Oo" 
    # headers = {"Authorization": f"Bearer {api_key}"}
    # payload = {"prompt": prompt}

    # response = requests.post(api_url, json=payload, headers=headers)
    # if response.status_code != 200:
    #     raise HTTPException(
    #         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    #         detail="Failed to generate description using Gemini API.",
    #     )
    # return response.json().get("text", "").strip()
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
    db: db_dependency,
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

