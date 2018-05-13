import cv2
import numpy as np

def crop(img, box, use_normalized_coordinates=False):
    """Returns image cropped according to box.

    Args:
    - img: numpy array representing image to crop
    - box: vector of shape (, 4) with box coordinates in order:
        x1, y1, x2, y2
    - use_normalized_coordinates: whether values in box are normalized
        or absolute coordinates
    """
    img_height, img_width = img.shape[:2]
    left, top, right, bottom = box

    if use_normalized_coordinates:
        left = int(left * img_width)
        right = int(right * img_width)
        top = int(top * img_height)
        bottom = int(bottom * img_height)

    return img[top:bottom, left:right]

def resize(img, imgDim):
    return cv2.resize(img, (imgDim, imgDim))

def filter_boxes_by_score(boxes, scores, threshold=0.7):
    indices = np.where(scores > threshold)
    filtered_boxes = boxes[indices]
    filtered_scores = scores[indices]
    return filtered_boxes, filtered_scores

def get_largest_bounding_box(boxes):
    """Returns the largest bounding box.

    Args:
    - boxes: numpy array of shape (None, 4) with box coordinates in order:
        x1, y1, x2, y2
    """
    box_sizes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_index = np.argmax(box_sizes)
    return boxes[largest_index]

def get_best_score_box(boxes, scores):
    """Returns the box with the highest confidence score.

    Args:
    - boxes: matrix of shape (None, 4) with box coordinates in order:
        x1, y1, x2, y2
    - scores: array of confidence scores.
    """
    highest_score_index = np.argmax(scores)
    return boxes[highest_score_index]
