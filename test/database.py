import os
import sqlite3
from deepface import DeepFace
import numpy as np
from PIL import Image

# Database setup
def initialize_database(db_name="face_recognition.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Function to insert embeddings
def add_embedding_to_db(name, embedding, db_name="face_recognition.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Convert embedding (numpy array) to string
    embedding_str = ",".join(map(str, embedding))
    cursor.execute("INSERT INTO face_embeddings (name, embedding) VALUES (?, ?)", (name, embedding_str))
    conn.commit()
    conn.close()

# Function to retrieve embeddings
def load_embeddings_from_db(db_name="face_recognition.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM face_embeddings")
    results = cursor.fetchall()
    conn.close()

    # Convert embeddings back to numpy arrays
    embeddings = {}
    for name, embedding_str in results:
        embedding = np.array(list(map(float, embedding_str.split(","))))
        embeddings[name] = embedding
    return embeddings

def process_images(name, image_folder, model_name="Facenet512", db_name="face_recognition.db"):
    initialize_database(db_name)
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)

        # Ensure it's an image file
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:

                # Preprocess image and extract embedding
                embedding = DeepFace.represent(img_path=file_path, model_name=model_name, enforce_detection=False)
                embedding_vector = embedding[0]["embedding"]

                # Add to database
                add_embedding_to_db(name, embedding_vector, db_name)
                print(f"Added {name} to the database.")

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")