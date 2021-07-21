# EMNIST_handwritten_recognition_api
### Test for the Machine Learning Engineer Position
### Author: Thomas Liu de Almeida
Flask based API with a Convolutional Keras Neural Network Trained with the EMINST-letters dataset, particularly with the MATLAB set.
For a first version, the case insensitive approach was chosen.
Server api on "api.py", responds a label given a image
Client testing on "consumer_api.py", requests a label for the image of a characters. Testing made with 8 handwritten character images, sorted with upper and lower cases.
