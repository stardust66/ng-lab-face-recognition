import cv2
import imageio
import os
import argparse
import tensorflow as tf
from src import model, align, utils, database

sess = tf.Session()
facenet = model.FaceNet(sess)
aligner = align.SSDAligner()

parser = argparse.ArgumentParser(description="Perform realtime facial "
                                             "recognition")
parser.add_argument("database_path", help="Path to the embeddings database")

def continous_detect(embeddings_database, names):
    with utils.VideoCapture(0) as capture:
        while True:
            success, image = capture.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face = aligner.align_and_crop_face(image)

            if face is None:
                print("No face detected")
                continue

            cv2.imshow("Face", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break

            embeddings = facenet.get_embeddings(face)
            person_index, distance = utils.classify(embeddings_database,
                                                    embeddings, 0.35)
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
    aligner.sess.close()
