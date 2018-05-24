from contextlib import contextmanager
import cv2
import numpy as np
from .facenet.src import facenet

def fixed_standardization(image):
    """Performs image standardization as suggested by David Sandberg

    Some models are trained using fixed image standardization instead of
    per image standardization. Consistency in this preprocessing step can
    really boost performance.
    """
    # https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L121
    return (image - 127.5) / 128.0

def per_image_standardization(image):
    """Performs per image standardization

    Uses the tensorflow implementation of capping the standard deviation away
    from zero to prevent dividing by zero in case of uniform images.
    """
    std = np.std(image)
    mean = np.mean(image)
    adjusted_std = max(std, 1.0 / np.sqrt(image.size))
    return (image - mean) / adjusted_std

def distance(embeddings1, embeddings2, metric="squared_l2"):
    """Calculates distance between embeddings

    Metric can be 'squared_l2' or 'cosine_similarity'
    """
    # In the facenet library, distance metric 0 is squared l2 norm while
    # distance metric 1 is cosine similarity.
    if metric == "squared_l2":
        return facenet.distance(embeddings1, embeddings2, distance_metric=0)[0]
    elif metric == "cosine_similarity":
        return facenet.distance(embeddings1, embeddings2, distance_metric=1)[0]
    else:
        raise ValueError("metric must be either squared_l2 or cosine_similarity"
                         ", instead received '{}'.".format(metric))

def distance_from_all(embeddings_database, embeddings, metric="squared_l2"):
    """Calculates the distance between embedding and all embeddings

    Metric can be 'squared_l2' or 'cosine_similarity'. The implementation
    is largely based on David Sandberg's facenet.py, just adapted to work
    with an array of embedding vectors.
    """
    if metric == "squared_l2":
        diff = np.subtract(embeddings_database, embeddings)
        norm = np.sum(np.square(diff), axis=-1)
        return np.squeeze(norm)
    elif metric == "cosine_similarity":
        dot = np.sum(np.multiply(embeddings_database, embeddings), axis=-1)
        database_norm = np.linalg.norm(embeddings_database, axis=-1)
        embeddings_norm = np.linalg.norm(embeddings)
        cosine = dot / (database_norm * embeddings_norm)
        return np.arccos(cosine) / np.pi
    else:
        raise ValueError("metric must be either squared_l2 or cosine_similarity"
                         ", instead received '{}'.".format(metric))

def classify(embeddings_database, embeddings, threshold, metric):
    distance = distance_from_all(embeddings_database, embeddings, metric)
    min_distance = np.amin(distance)

    if min_distance > threshold:
        return -1, min_distance

    person_index = np.argmin(distance)
    return person_index, min_distance

@contextmanager
def VideoCapture(camera_id):
    """Context manager wrapper for cv2.VideoCapture"""
    capture = cv2.VideoCapture(camera_id)
    try:
        yield capture
    finally:
        capture.release()
