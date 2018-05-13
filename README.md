# Engineering Lab Facial Recognition
Raspberry Pi that greets people at the entrance of the engineering lab using
facial recognition. Still in the preliminary stage.

## Dependencies
See `requirements.txt`. Many of the requirements are inherited from
https://github.com/davidsandberg/facenet.

## Setup
You need to have python 3 installed. This repository uses git-lfs, so you
should probably get it too, here: https://git-lfs.github.com/. You should also
download pretrained facenet models from
https://github.com/davidsandberg/facenet#pre-trained-models, unzip them, and
place them under the saved_models top level directory. Then run:
```
git clone --recursive https://github.com/StPauls-Computer-Science/ng-lab-face-recognition.git
pip install -r requirements.txt
```

## Notebook
In the notebook `Embeddings_Test.ipynb`, I've been exploring the embeddings
and building a simple face recognizer. If the GitHub notebook viewer doesn't
work, use jupyter's [nbviewer](https://nbviewer.jupyter.org/) and paste in the
notebook URL or click [here](https://nbviewer.jupyter.org/github/StPauls-Computer-Science/ng-lab-face-recognition/blob/master/Embeddings_Test.ipynb).

#### A Note About Test Images
I've provided three pictures of myself and Sam Henderson, who agreed to have
his pictures put online. For other pictures referenced in the notebook, just
find some pictures online or get some from Google Images.

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
is an open source face recognition module developed by Carnegie Mellon.
