def crop(img, box):
    return img[box[1]:box[3], box[0]:box[2]]
