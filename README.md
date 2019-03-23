# Project Turing
(**T**raining **U**nsettling **R**enders **I**n **N**eural **G**raphs)

This repository hosts the primary digital artifact from an investigation into phenomena of overfitting in convolutional neural networks.

### Demo
The **demo** directory contains a scaled-down version of the dataset and pre-configured demonstration of how image rotation and noise augmentation networks operate whilst training.
Results of the demo training are saved in he respective **demo-results** folder.

### Main Network
The core network structure, merged from two separate branches, can be found in the **binary-classifier-network.py** file, and its respective versions in general experiments branches.

High-level overview of the network structure can be seen below:
![Network Structure](results/network-model.png)

Network Summary:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 64, 64, 20)        1520
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 20)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 20)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 50)        25050
_________________________________________________________________
activation_2 (Activation)    (None, 32, 32, 50)        0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 16, 50)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 12800)             0
_________________________________________________________________
dense_1 (Dense)              (None, 50)                640050
_________________________________________________________________
activation_3 (Activation)    (None, 50)                0
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 102
_________________________________________________________________
activation_4 (Activation)    (None, 2)                 0
=================================================================
Total params: 666,722
Trainable params: 666,722
Non-trainable params: 0
_________________________________________________________________
```


