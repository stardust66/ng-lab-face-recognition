import cv2
import imageio
import os
import argparse
import tensorflow as tf
from src import model, align, utils, database
from src.recognizer import Recognizer

parser = argparse.ArgumentParser(description="Perform realtime facial "
                                             "recognition")
parser.add_argument("database_path", help="Path to the embeddings database")

model_path = os.path.join(
    os.path.dirname(__file__),
    "saved_models/model_vggface2"
)

def continous_detect(embeddings_database, names):
    recognizer = Recognizer(embeddings_database, model_path,
                            use_fixed_standardization=True,
                            metric="cosine_similarity")

    with utils.VideoCapture(0) as capture, recognizer:
        while True:
            success, image = capture.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            person_index, distance = recognizer.classify(image, 0.35,
                                                         debug=True)
            if person_index is None:
                print("No face detected")
                continue

            if person_index != -1:
                print(names[person_index])
                print("Distance: {}".format(distance))
            else:
                print("Unknown face")

if __name__ == "__main__":
    args = parser.parse_args()
    database_path = os.path.abspath(args.database_path)
    embeddings_database, names = database.load_database(database_path)
    continous_detect(embeddings_database, names)
