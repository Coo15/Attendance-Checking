import os
import sqlite3
from deepface import DeepFace
import numpy as np
from PIL import Image

# Initialize the database
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

# Function to add embedding to database
def add_embedding_to_db(name, embedding, db_name="face_recognition.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    embedding_str = ",".join(map(str, embedding))
    cursor.execute("INSERT INTO face_embeddings (name, embedding) VALUES (?, ?)", (name, embedding_str))
    conn.commit()
    conn.close()

# Function to process images and generate embeddings
def process_images(image_folder, model_name="Facenet512", db_name="face_recognition.db"):
    initialize_database(db_name)
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)

        # Ensure it's an image file
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Extract the name from the file (e.g., John_Doe.jpg -> John Doe)
                name = os.path.splitext(file_name)[0].replace("_", " ")

                # Preprocess image and extract embedding
                embedding = DeepFace.represent(img_path=file_path, model_name=model_name, enforce_detection=False)
                embedding_vector = embedding[0]["embedding"]

                # Add to database
                add_embedding_to_db(name, embedding_vector, db_name)
                print(f"Added {name} to the database.")

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

# Main function
def main():
    image_folder = "images"  # Change to the path where your images are stored
    db_name = "face_recognition.db"

    print("Processing images and adding embeddings to the database...")
    process_images(image_folder, model_name="Facenet512", db_name=db_name)
    print("All embeddings have been added to the database.")

if __name__ == "__main__":
    main()
