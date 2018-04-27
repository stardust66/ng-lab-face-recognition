import numpy as np
import tensorflow as tf
import facenet.src.facenet as facenet

def load_model():
    facenet.load_model("./saved_models/model_vggface2")

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
