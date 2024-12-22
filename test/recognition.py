import cv2
from mtcnn import MTCNN
from deepface import DeepFace
import numpy as np

# Import SQLite helper functions
from database import initialize_database, add_embedding_to_db, load_embeddings_from_db

# Initialize MTCNN detector
detector = MTCNN()

# Initialize SQLite database
db_name = "face_recognition.db"
initialize_database(db_name)

# Function to preprocess face and extract embeddings
def get_face_embedding(face, model_name="Facenet512"):
    try:
        embedding = DeepFace.represent(face, model_name=model_name, enforce_detection=False)
        return embedding[0]["embedding"]
    except Exception as e:
        print(f"Error in embedding generation: {e}")
        return None

def recognize_face(face_embedding, known_faces, threshold=20):
    recognized_name = "Unknown"
    min_distance = float("inf")

    for name, known_embedding in known_faces.items():
        distance = np.linalg.norm(np.array(face_embedding) - np.array(known_embedding))
        if distance < min_distance:
            min_distance = distance
            recognized_name = name

    if min_distance < threshold:
        return recognized_name, min_distance
    return "Unknown", min_distance

def main():
    global db_name

    # Load known embeddings from the database
    known_faces = load_embeddings_from_db(db_name)

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert frame to RGB (MTCNN requires RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = detector.detect_faces(frame_rgb)

        # Process each detected face
        for result in results:
            x, y, width, height = result['box']

            # Extract the face from the frame
            face = frame_rgb[y:y + height, x:x + width]

            # Get face embedding
            face_embedding = get_face_embedding(face)

            if face_embedding is not None:
                # Recognize the face
                name, confidence = recognize_face(face_embedding, known_faces)

                # Draw bounding box and name
                label = f"{name} ({confidence:.2f})"
                if name == "Unknown":
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

        # Show the frame
        cv2.imshow("Face Detection and Recognition", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
