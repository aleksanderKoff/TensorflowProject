import tensorflow as tf
from keras.layers.preprocessing.image_preprocessing import ResizeMethod
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential

print(tf.__version__)
mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.expand_dims(X_train, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)

print(X_train.shape)
print(X_test.shape)

import numpy as np
import matplotlib.pyplot as plt

X_train = tf.cast(
    X_train, dtype=tf.float32, name=None
)
X_test = tf.cast(
    X_test, dtype=tf.float32, name=None
)

X_train = X_train/255.0
X_test = X_test/255.0

print(np.min(X_train))
print(np.max(X_train))

plt.figure()
plt.imshow(X_train[3])
plt.colorbar()

model = tf.keras.models.Sequential([
tf.keras.layers.Resizing(
    32, 32, interpolation='bilinear', crop_to_aspect_ratio=False,
),
 tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights=None, input_tensor=None,
    input_shape=(32, 32, 1)
    ),
    tf.keras.layers.Dense(10)
])


model.compile(optimizer= "adam", loss= "sparse_categorical_crossentropy", metrics= ["accuracy"])
model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_loss)
print(test_acc)

model.save('my_model.h5')
