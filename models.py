from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import JSON, Column, Integer, String 
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key = True)
    name = Column(String, nullable = False)
    embedding = Column(JSON, nullable = False)

    def __repr__(self) -> str:
        return f"<User name = {self.name}>"
