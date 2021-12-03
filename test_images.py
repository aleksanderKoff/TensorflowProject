import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

images = ['images/test_1.jpg', 'images/test_2.jpg', 'images/test_3.jpg',
          'images/test_4.jpg', 'images/test_5.jpg', 'images/test_6.jpg',
          'images/test_7.jpg', 'images/test_8.jpg', 'images/test_9.jpg']


def load_image(filename):
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = PIL.ImageOps.invert(img)
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    return img


def run_example():
    for i in images:
        img = load_image(i)
        model = tf.keras.models.load_model('my_model_mnist.h5')
        predict_value = model.predict(img)
        digit = argmax(predict_value)
        print(digit)

run_example()
