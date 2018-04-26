"""Logs pretrained model for tensorboard graph visualization"""

import tensorflow as tf
import facenet.src.facenet as facenet

with tf.Session() as sess:
    facenet.load_model("./saved_models/model_casia")
    writer = tf.summary.FileWriter('./logs/facenet-pretrained-log',
                                   sess.graph)
