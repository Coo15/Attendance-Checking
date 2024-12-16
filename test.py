import sqlite3

# Step 1: Connect to the database (or create it if it doesn't exist)
connection = sqlite3.connect("example.db")
cursor = connection.cursor()

# Step 2: Create a table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL
)
""")
connection.commit()

# Step 3: CRUD Operations
# 1. Create
def create_user(name, email):
    cursor.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    connection.commit()
    print(f"User {name} added successfully.")

# 2. Read
def read_users():
    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()
    for user in users:
        print(user)

# 3. Update
def update_user(user_id, new_name, new_email):
    cursor.execute("UPDATE users SET name = ?, email = ? WHERE id = ?", (new_name, new_email, user_id))
    connection.commit()
    print(f"User {user_id} updated successfully.")

# 4. Delete
def delete_user(user_id):
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    connection.commit()
    print(f"User {user_id} deleted successfully.")

# Step 4: Test the CRUD Functions
if __name__ == "__main__":
    # Create
    create_user("Alice", "alice@example.com")
    create_user("Bob", "bob@example.com")
    
    # Read
    print("Users in the database:")
    read_users()
    
    # Update
    update_user(1, "Alice Smith", "alice.smith@example.com")
    
    # Read after update
    print("After update:")
    read_users()
    
    # Delete
    delete_user(2)
    
    # Read after delete
    print("After delete:")
    read_users()

# Close the connection
connection.close()
