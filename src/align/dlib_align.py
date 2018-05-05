import os
from .openface import AlignDlib

class DlibAligner():
    def __init__(self):
        aligner_path = os.path.join(
            os.path.dirname(__file__),
            "./openface/models/dlib/shape_predictor_68_face_landmarks.dat"
        )
        self.aligner = AlignDlib(aligner_path)

    def align_and_crop_face(self, image, image_dim=160):
        """Transform and align a face in an image

        Args:
        - image: image to process
        - image_dim: the edge length in pixels of the square the image is
        resized to.
        """
        return self.aligner.align(image_dim, image)
