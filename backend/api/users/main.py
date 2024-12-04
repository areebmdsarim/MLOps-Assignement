from fastapi import FastAPI, status, Depends, HTTPException
from backend.db.models import users # models
from backend.db.models.base import engine, SessionLocal #database; which has engine and session local for sqllite
from typing import Annotated
from sqlalchemy.orm import Session
from backend.api.users import auth
from backend.api.users.auth import get_current_user
from backend.swagger import router as swagger_router
from backend.db.models.base import Base  # Import Base directly


from backend.api.users.auth import router as auth_router
from backend.api.users.data_source import router as data_source_router
from backend.db.models.data_source import DataSource, Database, Table, Column


app = FastAPI()
app.include_router(auth.router)
app.include_router(swagger_router)


app.include_router(auth_router)
app.include_router(data_source_router)

# users.Base.metadata.create_all(bind=engine) #models.Base
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

@app.get("/", status_code = status.HTTP_200_OK)
async def user(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    return {"User": user}

