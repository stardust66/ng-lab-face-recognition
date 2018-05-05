import numpy as np
import tensorflow as tf
from ..facenet.src.align import detect_face
from .utils import crop, resize

class MtcnnAligner():
    def __init__(self):
        sess = tf.get_default_session()
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess,
                                                                   None)

    def align_and_crop_face(self, image, image_dim=160):
        """Aligns and crops face with MTCNN aligner

        Args:
        - image: image to align.
        - image_dim: side length of image to resize to.
        """

        minsize = 20
        threshold = [0.6, 0.6, 0.7]
        factor = 0.709
        margin = 32
        bounding_boxes, _ = detect_face.detect_face(image, minsize, self.pnet,
                                                    self.rnet, self.onet,
                                                    threshold, factor)
        bounding_boxes = bounding_boxes[:, 0:4]

        num_faces = bounding_boxes.shape[0]
        if num_faces < 1:
            return []

        if num_faces == 1:
            filtered_box = np.squeeze(bounding_boxes)

        # Weight face in the center more heavily
        bb_x1, bb_y1, bb_x2, bb_y2 = np.split(bounding_boxes, 4, 1)
        bounding_box_width = bb_x2 - bb_x1
        bounding_box_height = bb_y2 - bb_y1
        bounding_box_size = bounding_box_width * bounding_box_height

        image_size = np.asarray(image.shape[0:2])
        image_center = image_size / 2

        offsets = np.vstack([(bb_x1 + bb_x2) / 2 - image_center[0],
                             (bb_y1 + bb_y2) / 2 - image_center[1]])
        offset_distance_squared = np.sum(np.power(offsets, 2), axis=0)
        offset_scaled = offset_distance_squared * 2
        prominent_bounding_box_index = np.argmax(bounding_box_size
                                                 - offset_scaled)
        prominent_bounding_box = bounding_boxes[prominent_bounding_box_index, :]
        filtered_box = np.squeeze(prominent_bounding_box)

        output_box = np.zeros(4, dtype=np.int32)
        output_box[0] = np.maximum(filtered_box[0] - margin/2, 0)
        output_box[1] = np.maximum(filtered_box[1] - margin/2, 0)
        output_box[2] = np.minimum(filtered_box[2] + margin/2, image_size[1])
        output_box[3] = np.minimum(filtered_box[3] + margin/2, image_size[0])

        return resize(crop(image, output_box), image_dim)
