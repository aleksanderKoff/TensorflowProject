import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

X_train = X_train/255.0
X_test = X_test/255.0

print(np.min(X_train))
print(np.max(X_train))

plt.figure()
plt.imshow(X_train[3])
plt.colorbar()
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer= "adam", loss= "sparse_categorical_crossentropy", metrics= ["accuracy"])
model.fit(X_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(test_loss)
print(test_acc)

model.save('my_model_mnist.h5')
