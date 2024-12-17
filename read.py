from main import session
from models import User
from sqlalchemy import select

users = session.query(User).all()
for user in users:
    print(user)

result = session.query(User).filter_by(
    embedding = get_embedding()
)
print(result)

