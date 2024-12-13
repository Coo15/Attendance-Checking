from os import listdir
from os.path import isdir
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    """
    Extracts and aligns a single face from an image file.

    Parameters:
        filename (str): Path to the image file.
        required_size (tuple): Target size for the output face image.

    Returns:
        np.ndarray: Extracted face image as an array of the required size, or None if no face is detected.
    """
    try:
        # Load image using OpenCV for better performance
        image = cv2.imread(filename)
        if image is None:
            print(f"Failed to load image: {filename}")
            return None

        # Convert to RGB format (OpenCV loads in BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create the MTCNN detector
        detector = MTCNN()

        # Detect faces in the image
        results = detector.detect_faces(image)
        if len(results) == 0:
            print(f"No face detected in {filename}")
            return None

        # Select the face with the highest confidence score
        best_result = max(results, key=lambda res: res['confidence'])

        # Extract the bounding box
        x1, y1, width, height = best_result['box']
        x1, y1 = max(0, x1), max(0, y1)  # Ensure coordinates are within bounds
        x2, y2 = x1 + width, y1 + height

        # Extract the face region
        face = image[y1:y2, x1:x2]

        # Resize face to the model's required size
        face_image = Image.fromarray(face)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)

        return face_array

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        if face is None:
            continue
        # store
        faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

# load train dataset
X_train, Y_train = load_dataset('./data/train/')
print(X_train.shape, Y_train.shape)
# load test dataset
X_test, Y_test = load_dataset('./data/val/')
# save arrays to one file in compressed format
savez_compressed('facenet/faces_detection.npz', X_train, Y_train, X_test, Y_test)