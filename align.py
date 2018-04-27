import numpy as np
import tensorflow as tf
import facenet.src.align.detect_face as detect_face

class Aligner():
    def __init__(self):
        sess = tf.get_default_session()
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess,
                                                                       None)
    
    def align_face(self, img, minsize=20, threshold=[0.6, 0.6, 0.7],
                          factor=0.709, margin=32):

        bounding_boxes, _ = detect_face.detect_face(img, minsize, self.pnet,
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

        img_size = np.asarray(img.shape[0:2])
        img_center = img_size / 2

        offsets = np.vstack([(bb_x1 + bb_x2) / 2 - img_center[0],
                             (bb_y1 + bb_y2) / 2 - img_center[1]])
        offset_distance_squared = np.sum(np.power(offsets, 2), axis=0)
        offset_scaled = offset_distance_squared * 2
        prominent_bounding_box_index = np.argmax(bounding_box_size
                                                 - offset_scaled)
        prominent_bounding_box = bounding_boxes[prominent_bounding_box_index, :]
        filtered_box = np.squeeze(prominent_bounding_box)

        output_box = np.zeros(4, dtype=np.int32)
        output_box[0] = np.maximum(filtered_box[0] - margin/2, 0)
        output_box[1] = np.maximum(filtered_box[1] - margin/2, 0)
        output_box[2] = np.minimum(filtered_box[2] + margin/2, img_size[1])
        output_box[3] = np.minimum(filtered_box[3] + margin/2, img_size[0])

        return output_box

def crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]
