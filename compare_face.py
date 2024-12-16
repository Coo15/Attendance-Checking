import json
from credb import session
import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def compare_face():
    # Capture the face and get its embedding
    embedding = get_embedding()
    if embedding is None:
        '''No embedding captured for comparison'''
        return

    # Retrieve all saved embeddings from the database
    users = session.query(User).all()
    if not users:
        '''No users in the database for comparison'''
        return

    # Convert the embedding to NumPy for comparison
    embedding_np = np.array(embedding)

    # Compare with each stored user
    matched_user = None
    highest_similarity = 0.0

    for user in users:
        stored_embedding = np.array(json.loads(user.embedding))
        similarity = cosine_similarity(embedding_np, stored_embedding)
        print(f"Comparing with {user.name}: Similarity = {similarity:.2f}")
        if similarity > highest_similarity and similarity > 0.5:  # Threshold
            highest_similarity = similarity
            matched_user = user

    if matched_user:
        print(f"Matched with {matched_user.name} (Similarity = {highest_similarity:.2f})")
    else:
        print("No match found.")
