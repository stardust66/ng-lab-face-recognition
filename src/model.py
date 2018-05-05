import os
import numpy as np
import tensorflow as tf
from .facenet.src import facenet

def load_model():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "saved_models/model_vggface2"
    )
    facenet.load_model(model_path)

def get_embeddings(img):
    sess = tf.get_default_session()
    graph = sess.graph

    # Convert img shape (160, 160, 3) to (1, 160, 160, 3)
    img = np.expand_dims(img, axis=0)

    embeddings = graph.get_tensor_by_name("embeddings:0")
    input_images = graph.get_tensor_by_name("input:0")
    input_phase_train = graph.get_tensor_by_name("phase_train:0")

    return sess.run(embeddings, feed_dict={
        input_images: img,
        input_phase_train: False
    })
