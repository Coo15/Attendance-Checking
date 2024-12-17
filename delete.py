from main import session
from models import User

user = session.query(User).filter_by(
    id = 1
)

session.delete(user)