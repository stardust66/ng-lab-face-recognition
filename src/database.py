import tensorflow as tf
import numpy as np
import imageio
import os
from . import align
from . import model

def create_database(photo_paths, output_path):
    """Creates embeddings database for a list of photos.

    This uses the names of the photos as the names for the people.
    Args:
    - photo_paths: a list of paths to photos
    - output_path: the path of the output .npz file
    """
    sess = tf.Session()
    facenet = model.FaceNet(sess)
    aligner = align.SSDAligner()

    num_photos = len(photo_paths)
    embeddings = np.zeros((num_photos, 512))
    names = []

    for i, photo_path in enumerate(photo_paths):
        photo = imageio.imread(photo_path)
        face = aligner.align_and_crop_face(photo)

        name = os.path.basename(photo_path).split(".")[0]
        names.append(name)

        photo_embeddings = facenet.get_embeddings(face)
        embeddings[i] = photo_embeddings

    np.savez(output_path, embeddings=embeddings, names=names)

    aligner.sess.close()
    sess.close()

def load_database(database_path):
    loaded = np.load(database_path)
    embeddings = loaded["embeddings"]
    names = loaded["names"]
    return embeddings, names
