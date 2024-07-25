from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from datetime import datetime
from typing import List, Optional, Union
from sqlalchemy.orm import Session
from models import User as DbUser
# from dependencies import get_db
from fastapi.security import OAuth2PasswordBearer
from models import get_conn, get_db_conn
from utils import (
    ALGORITHM,
    JWT_SECRET_KEY
)

reuseable_oauth = OAuth2PasswordBearer(
    tokenUrl="/login",
    scheme_name="JWT"
)

class TokenPayload(BaseModel):
    sub: str
    exp: int
    iat: Optional[int] = None
    scopes: Optional[List[str]] = None


class SystemUser(BaseModel):
    email: str
    password:str
    id:str |None

async def get_current_user(token: str = Depends(reuseable_oauth)) -> SystemUser:
    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        if datetime.fromtimestamp(token_data.exp) < datetime.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # user = db.query(DbUser).filter(DbUser.id == token_data.sub).first()
    conn = get_db_conn()
    if conn is None:
        print("No connection")
        conn = get_conn()
    
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Persons WHERE FirstName = ?", token_data.sub)
    row = cursor.fetchone()

    
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",

        )
        
    return SystemUser(id=row.UserID, email=row.FirstName, password=row.LastName)