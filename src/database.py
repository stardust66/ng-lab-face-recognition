import tensorflow as tf
import numpy as np
import imageio
import os
from . import align, model, utils

def create_database(photo_paths, output_path, model_path,
                    use_fixed_standardization=False):
    """Creates embeddings database for a list of photos.

    This uses the names of the photos as the names for the people.
    Args:
    - photo_paths: a list of paths to photos
    - output_path: the path of the output .npz file
    - use_fixed_standardization: Whether to use fixed image standardization
    described by David Sandberg
    """
    sess = tf.Session()
    facenet = model.FaceNet(sess, model_path)
    aligner = align.SSDAligner()

    num_photos = len(photo_paths)
    embeddings = np.zeros((num_photos, 512))
    names = []

    print("Building embeddings database...")
    print("0.00% =>", end="")

    for i, photo_path in enumerate(photo_paths):
        photo = imageio.imread(photo_path)
        face = aligner.align_and_crop_face(photo)
        name = os.path.basename(photo_path).split(".")[0]

        if face is None:
            print()
            print("Couldn't align photo for {}, skipping".format(name))
            continue

        names.append(name)

        # Perform standardization
        if use_fixed_standardization:
            face = utils.fixed_standardization(face)
        else:
            face = utils.per_image_standardization(face)

        photo_embeddings = facenet.get_embeddings(face)
        embeddings[i] = photo_embeddings

        percentage_done = i / num_photos * 100
        progress_bar = "=" * int(percentage_done / 2) + "=>"
        print("\r{:.2f}% {}".format(percentage_done, progress_bar), end="")

    print("\r100.00% {}".format("=" * 50 + "=>"))
    np.savez(output_path, embeddings=embeddings, names=names)

    aligner.sess.close()
    sess.close()

def load_database(database_path):
    loaded = np.load(database_path)
    embeddings = loaded["embeddings"]
    names = loaded["names"]
    return embeddings, names
