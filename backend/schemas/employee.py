from pydantic import BaseModel
from typing import Optional
from uuid import UUID

class EmployeeBase(BaseModel):
    name: str
    address: Optional[str] = None

class EmployeeCreate(EmployeeBase):
    pass

class EmployeeRead(EmployeeBase):
    employee_id: int

    class Config:
        orm_mode = True
