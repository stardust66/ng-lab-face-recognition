# Engineering Lab Facial Recognition
Recognize different people's faces and greet them verbally with just a single
picture of each person. Works well with bad lighting and non-frontal poses.

## Setup
- Install python3
- Install [git-lfs](https://git-lfs.github.com)
- Download [pretrained models](https://github.com/davidsandberg/facenet#pre-trained-models)
and put them in the `saved_models` top level directory
- Install `mpg321` with a package manager or [here](https://sourceforge.net/projects/mpg321/files/latest/download?source=files)
- Run
```
git clone --recursive https://github.com/StPauls-Computer-Science/ng-lab-face-recognition.git
pip install -r requirements.txt
```

## Running Recognition
Find a single picture of everyone you want to recognize and put them in a
folder. Then, generate a database of embeddings with
```
python create_database.py --use-fixed-standardization (output_file) \
    (input_directory)
```
Then, you can do real-time camera facial recognition by running
```
python camera_recognize.py (database_path)
```
where `database_path` refers to the file you just generated. The filenames will
be used as labels (people's names).

## Notebook
In the notebook `Facial Recognition with FaceNet.ipynb`, I've been exploring
the embeddings and building a simple face recognizer. Check it out for a brief
explanation of FaceNet and the embeddings.

#### A Note About Test Images
I've provided three pictures of myself and Sam Henderson, who agreed to have
his pictures put online. For other pictures referenced in the notebook, get
some pictures from Google Images.

## Visualizing the saved model
```
python log_saved.py
tensorboard --logdir logs/facenet-pretrained-log
```
Will give you cool graph visualizations in tensorboard so you can explore
the model visually.

<img src="https://i.imgur.com/N2HBm2d.png" width=800
alt="tensoboard visualizations">

## Acknowledgements
I'm using David Sandberg's implementation of FaceNet (included as a submodule)
and his pretrained weights. I'm also using OpenFace's wrapper around dlib's
face detection and alignment. [OpenFace](http://cmusatyalab.github.io/openface/)
is an open source face recognition module developed by Carnegie Mellon. To find
out more about FaceNet, read the original paper on arxiv [here](https://arxiv.org/abs/1503.03832)
