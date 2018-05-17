from contextlib import contextmanager
import cv2
import numpy as np
from . import facenet

def distance(embeddings1, embeddings2, metric="l2_norm"):
    """Calculates distance between embeddings

    Metric can be 'l2_norm' or 'cosine_similarity'
    """

    # In the facenet library, distance metric 0 is l2norm while distance
    # metric 1 is cosine similarity.
    if metric == "l2_norm":
        # Implementing L2 Norm myself because the facenet.distance doesn't take
        # square root after summing the differences. In practice this doesn't
        # matter as long as we are consistent, but I don't want to be confusing
        return np.linalg.norm(embeddings2 - embeddings1)
    elif metric == "cosine_similarity":
        return facenet.distance(embeddings1, embeddings2, distance_metric=1)[0]
    else:
        raise ("metric must be either l2_norm or cosine_similarity"
               ", instead received '{}'.".format(metric))

def distance_from_all(embeddings_database, embeddings):
    diff = np.subtract(embeddings_database, embeddings)
    norm = np.sqrt(np.sum(np.square(diff), axis=-1))
    return np.squeeze(norm)

def classify(embeddings_database, embeddings, threshold):
    distance = distance_from_all(embeddings_database, embeddings)
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
