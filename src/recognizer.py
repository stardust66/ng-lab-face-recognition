import cv2
import tensorflow as tf
from . import model, align, utils, database

class Recognizer():
    def __init__(self, database):
        self.sess = tf.Session()
        self.facenet = model.FaceNet(self.sess)
        self.aligner = align.SSDAligner()

        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, traceback):
        self.sess.close()
        self.aligner.sess.close()

    def classify(self, image, threshold=0.15, debug=False):
        """Identify a person from an image

        Returns index in the embeddings array.
        """
        face = self.aligner.align_and_crop_face(image)

        if face is None:
            return None, -1

        if debug:
            cv2.imshow("Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xff

        embeddings = self.facenet.get_embeddings(face)
        return utils.classify(self.database, embeddings, threshold)
