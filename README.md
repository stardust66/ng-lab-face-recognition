# Engineering Lab Facial Recognition
Raspberry Pi that greets people at the entrance of the engineering lab using
facial recognition. Still in the preliminary stage.

## Dependencies
See `requirements.txt`. Many of the requirements are inherited from
https://github.com/davidsandberg/facenet.

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
