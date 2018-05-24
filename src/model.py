import os
import numpy as np
import tensorflow as tf
from .facenet.src import facenet

class FaceNet():
    def __init__(self, sess, model_path):
        self.sess = sess

        with self.sess.as_default():
            facenet.load_model(model_path)

    def get_embeddings(self, img):
        graph = self.sess.graph

        # Convert img shape (160, 160, 3) to (1, 160, 160, 3)
        img = np.expand_dims(img, axis=0)

        embeddings = graph.get_tensor_by_name("embeddings:0")
        input_images = graph.get_tensor_by_name("input:0")
        input_phase_train = graph.get_tensor_by_name("phase_train:0")

        return self.sess.run(embeddings, feed_dict={
            input_images: img,
            input_phase_train: False
        })
