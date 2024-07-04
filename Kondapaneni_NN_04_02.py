import pytest
import numpy as np
from Kondapaneni_04_01 import CNN
import os
import tensorflow as tf


def test_train():
  (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
  X_train = (X_train / 255.0 - 0.5).astype(np.float32)
  Y_train = (Y_train / 255.0 - 0.5).astype(np.float32)
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
  Y_train = tf.keras.utils.to_categorical(Y_train, 10)
  X_train = X_train[:100]
  Y_train = Y_train[:100]
  my_cnn = CNN()
  my_cnn.set_loss_function(loss="CategoricalCrossentropy")
  my_cnn.add_input_layer(shape=(28,28,1), name="input0")
  my_cnn.append_conv2d_layer(8, (3, 3), activation='relu',name="test_append_conv2d_layer_1")
  my_cnn.append_conv2d_layer(16, (3, 3), activation='relu',name="test_append_conv2d_layer_2")
  my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling_1')
  my_cnn.append_conv2d_layer(32, (3, 3), activation='relu',name="test_append_conv2d_layer_3")
  my_cnn.append_conv2d_layer(64, (3, 3), activation='relu',name="test_append_conv2d_layer_4")
  my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling_2')
  my_cnn.append_flatten_layer(name='flatten_1')
  my_cnn.append_dense_layer(num_nodes=512, activation='linear', name="layer1")
  my_cnn.append_dense_layer(num_nodes=10, activation='softmax', name="layer2")
  loss_values,accuracy_values = my_cnn.train(X_train,Y_train,batch_size = 32,num_epochs = 2)
  assert loss_values[1] < loss_values[0]
  assert accuracy_values[0] < accuracy_values[1]
  assert accuracy_values[0] > 0.5

def test_evaluate():
  (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
  X_train = (X_train / 255.0 - 0.5).astype(np.float32)
  Y_train = (Y_train / 255.0 - 0.5).astype(np.float32)
  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
  Y_train = tf.keras.utils.to_categorical(Y_train, 10)
  X_train = X_train[:100]
  Y_train = Y_train[:100]
  my_cnn = CNN()
  my_cnn.set_loss_function(loss="CategoricalCrossentropy")
  my_cnn.add_input_layer(shape=(28,28,1), name="input0")
  my_cnn.append_conv2d_layer(8, (3, 3), activation='relu',name="test_append_conv2d_layer_1")
  my_cnn.append_conv2d_layer(16, (3, 3), activation='relu',name="test_append_conv2d_layer_2")
  my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling_1')
  my_cnn.append_conv2d_layer(32, (3, 3), activation='relu',name="test_append_conv2d_layer_3")
  my_cnn.append_conv2d_layer(64, (3, 3), activation='relu',name="test_append_conv2d_layer_4")
  my_cnn.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling_2')
  my_cnn.append_flatten_layer(name='flatten_1')
  my_cnn.append_dense_layer(num_nodes=512, activation='linear', name="layer1")
  my_cnn.append_dense_layer(num_nodes=10, activation='softmax', name="layer2")
  loss_values,accuracy_values = my_cnn.train(X_train,Y_train,batch_size = 32,num_epochs = 2)
  assert loss_values[1] < loss_values[0]
  assert accuracy_values[0] < accuracy_values[1]
  assert accuracy_values[0] > 0.5
