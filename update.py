from main import session
from models import User

update = session.query(User).filter_by(
    id = 1
)

User.name = ""
User.embedding = ""

session.commit()