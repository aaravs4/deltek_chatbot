from fastapi import Depends, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel, ValidationError
from datetime import datetime
from typing import List, Optional, Union
from fastapi.security import OAuth2PasswordBearer
import sqlite3
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
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users_new WHERE username = ?", (token_data.sub,))
    row = cursor.fetchone()

    
    if row is None:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",

        )
    conn.close()
        
    return SystemUser(id=row[2], email=row[0], password=row[1])