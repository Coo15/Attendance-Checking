from sqlalchemy import create_engine, Column, Integer, String, ARRAY
from sqlalchemy.orm import declarative_base, sessionmaker

#database setup
Base = declarative_base()
engine = create_engine("sqlite:///faces.db", echo=True)
Session = sessionmaker(bind=engine)
session = Session()

#database model
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    embedding = Column(ARRAY(float), unique=True)

#create the table
Base.metadata.create_all(engine)
