import os
import numpy as np
import tensorflow as tf
from .utils import get_best_score_box, crop, resize

class SSDAligner():
    def __init__(self):
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "models"
            "frozen_inference_graph_face.pb"
        )

        with tf.Graph().as_default() as self.graph:
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(checkpoint_path, "rb") as graph_file:
                serialized_graph = graph_file.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')

            boxes = self.graph.get_tensor_by_name("detection_boxes:0")
            scores = self.graph.get_tensor_by_name("detection_scores:0")

            boxes = tf.squeeze(boxes, name="final_boxes")
            scores = tf.squeeze(scores, name="final_scores")

        self.sess = tf.Session(graph=self.graph)

    def align_and_crop_face(self, image, image_dim=160):
        image_expanded = np.expand_dims(image, axis=0)
        image_tensor = self.graph.get_tensor_by_name("image_tensor:0")
        boxes = self.graph.get_tensor_by_name("final_boxes:0")
        scores = self.graph.get_tensor_by_name("final_scores:0")

        boxes, scores =  self.sess.run([boxes, scores], feed_dict={
            image_tensor: image_expanded
        })

        # Convert from y1, x1, y2, x2 to x1, y1, x2, y2
        boxes = boxes[:, [1, 0, 3, 2]]

        highest_score_box = get_best_score_box(boxes, scores)
        return resize(crop(image, highest_score_box,
                           use_normalized_coordinates=True), image_dim)
