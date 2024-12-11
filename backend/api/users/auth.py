from datetime import timedelta, datetime, timezone
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from jwt.exceptions import InvalidTokenError
from starlette import status
from backend.db.models.base import Postgresql_SessionLocal, MySQL_SessionLocal  # database
from backend.db.models.users import Users  # models
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from backend.schemas.schema import CreateUserRequest, Token, TokenData
from fastapi.responses import JSONResponse


router = APIRouter(
    prefix='/auth',
    tags=['auth']
)

blacklisted_tokens = set()

SECRET_KEY = "6VfDpFnWZXQGlkOQjkx6hsUgkRXDHyNCT2Fd7BcOGU8"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

def get_Postgresql_db():
    db = Postgresql_SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_MySQL_db():
    db = MySQL_SessionLocal()
    try:
        yield db
    finally:
        db.close()

Postgresql_db_dependency = Annotated[Session, Depends(get_Postgresql_db)]
MySQL_db_dependency = Annotated[Session, Depends(get_MySQL_db)]

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(db: Postgresql_db_dependency, create_user_request: CreateUserRequest):
    create_user_model = Users(
        username=create_user_request.username,
        hashed_password=bcrypt_context.hash(create_user_request.password),
        disabled=True,
    )
    db.add(create_user_model)
    db.commit()
    return {"message": "User created successfully"}

@router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Postgresql_db_dependency  # Pass db session here
) -> Token:
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)], db: Postgresql_db_dependency):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    await verify_token_blacklist(token)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[Users, Depends(get_current_user)],
):
    if current_user.disabled == False:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@router.post("/logout")
async def logout(request: Request, token: Annotated[str, Depends(oauth2_bearer)], current_user: Annotated[Users, Depends(get_current_user)]):
    """
    Logout the user by invalidating the token.
    """
    blacklisted_tokens.add(token)
    current_user.disabled = False
    return JSONResponse(content={"message": "Successfully logged out"}, status_code=status.HTTP_200_OK)

async def verify_token_blacklist(token: str):
    """
    Check if the token is blacklisted.
    """
    if token in blacklisted_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

class UserResponse(BaseModel):
    username: str
    disabled: bool

    class Config:
        orm_mode = True


def verify_password(plain_password, hashed_password):
    return bcrypt_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return bcrypt_context.hash(password)

def get_user(db: Session, username: str):
    return db.query(Users).filter(Users.username == username).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
