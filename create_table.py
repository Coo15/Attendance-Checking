from models import Base, User
from connect import engine

print("Create table...")
Base.metadata.create_all(bind=engine )