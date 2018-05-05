import cv2

def crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]

def resize(img, imgDim):
    return cv2.resize(img, (imgDim, imgDim))
