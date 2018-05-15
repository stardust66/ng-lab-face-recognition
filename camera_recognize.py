import cv2
import imageio
import os
import tensorflow as tf
from src import model, align, utils, database

sess = tf.Session()
facenet = model.FaceNet(sess)
aligner = align.SSDAligner()

def build_embedding_database():
    database_path = os.path.abspath("./databases/test.npz")
    photo_paths = [
        os.path.abspath("./test_images/sam.jpg"),
        os.path.abspath("./test_images/jason.jpg")
    ]
    if not os.path.isfile(database_path):
        database.create_database(photo_paths, database_path)

    return database.load_database(database_path)

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
            person_index = utils.classify(embeddings_database, embeddings, 0.3)
            if person_index != -1:
                print(names[person_index])
            else:
                print("Unknown face")

if __name__ == "__main__":
    embeddings_database, names = build_embedding_database()
    continous_detect(embeddings_database, names)
    aligner.sess.close()
