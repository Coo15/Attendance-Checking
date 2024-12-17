from models import User
from main import session


user1 = User(
    name = 'Vu Duc Thang',
    embedding = get_embedding()
)
session.add_all([user1])
session.commit()