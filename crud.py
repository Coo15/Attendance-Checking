from credb import session

def create_user(name):
    embedding = get_embedding()
    if embedding is not None:
        try:
            # Convert embedding to JSON string
            embedding_json = json.dumps(embedding.tolist())
            # Add to database
            new_user = User(name=name, embedding=embedding_json)
            session.add(new_user)
            session.commit()
            print(f"User {name} added successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error adding user: {e}")
    else:
        print("No face detected or embedding generated.")

def read_users():
    try:
        users = session.query(User).all()
        if not users:
            print("No users found.")
        else:
            print("Users in the database:")
            for user in users:
                print(f"ID: {user.id}, Name: {user.name}")
    except Exception as e:
        print(f"Error reading users: {e}")

def update_user(user_id, new_name=None):
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            if new_name:
                user.name = new_name
            
            # Optionally update embedding
            print("Do you want to capture a new embedding? (y/n)")
            if input().lower() == 'y':
                new_embedding = get_embedding()
                if new_embedding is not None:
                    user.embedding = json.dumps(new_embedding.tolist())
                    print("Embedding updated.")
            
            session.commit()
            print(f"User {user_id} updated successfully.")
        else:
            print("User not found.")
    except Exception as e:
        session.rollback()
        print(f"Error updating user: {e}")

def delete_user(user_id):
    try:
        user = session.query(User).filter_by(id=user_id).first()
        if user:
            session.delete(user)
            session.commit()
            print(f"User {user_id} deleted successfully.")
        else:
            print("User not found.")
    except Exception as e:
        session.rollback()
        print(f"Error deleting user: {e}")

