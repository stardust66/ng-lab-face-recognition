# ng-lab-face-recognition
Raspberry Pi that greets people at the entrance of the engineering lab using
facial recognition.

## Dependencies
See `requirements.txt`. Many of the requirements are inherited from
https://github.com/davidsandberg/facenet

## Visualizing the saved model
```
python log_saved.py
tensorboard --logdir logs/facenet-pretrained-log
```
Will give you cool graph visualizations in tensorboard so you can explore
the model visually.

<img src="https://i.imgur.com/N2HBm2d.png" width=800
alt="tensoboard visualizations">
