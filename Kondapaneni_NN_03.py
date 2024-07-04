import tensorflow as tf
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from keras.api._v2.keras import regularizers

def confusion_matrix(y_true, y_pred, n_classes=10):                                                     #confusion matrix function
  cm = np.zeros([n_classes,n_classes])
  for i in range(len(y_true)):                                                                          #calculating confusion matrix 
      cm[y_true[i],y_pred[i]] = cm[y_true[i],y_pred[i]] + 1 
  plt.matshow(cm, interpolation='nearest')                                                              #plotting confusion matrix
  plt.savefig("confusion_matrix.png")                                                                   #saving confusion matrix to confusion_matrix.png
  return cm

def split_data(X_train, Y_train, split_range=[0.8, 1.0]):                                               #code to split training data into train data and validation data
    start = int(split_range[0] * X_train.shape[0])                                                      #derived from helper codes given for assignment 02
    end = int(split_range[1] * X_train.shape[0])
    return np.concatenate((X_train[:start], X_train[end:])), np.concatenate(
        (Y_train[:start], Y_train[end:])), X_train[start:end], Y_train[start:end]


def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):                           #function to build network
  tf.keras.utils.set_random_seed(5368)
  history = []
  cm = []
  y_pred = []
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(8, (3, 3),strides=(1, 1),padding="same",kernel_regularizer= regularizers.L2(l2 = 0.0001), activation='relu', input_shape=(28, 28, 1)))
  model.add(tf.keras.layers.Conv2D(16, (3, 3),strides=(1, 1),padding="same",kernel_regularizer= regularizers.L2(l2 = 0.0001), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2),strides=(2, 2)))
  model.add(tf.keras.layers.Conv2D(32, (3, 3),strides=(1, 1),padding="same",kernel_regularizer= regularizers.L2(l2 = 0.0001), activation='relu'))
  model.add(tf.keras.layers.Conv2D(64, (3, 3),strides=(1, 1),padding="same",kernel_regularizer= regularizers.L2(l2 = 0.0001), activation='relu'))
  model.add(tf.keras.layers.MaxPooling2D((2, 2),strides=(2, 2)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512,kernel_regularizer= regularizers.L2(l2 = 0.0001), activation='relu'))
  model.add(tf.keras.layers.Dense(10,kernel_regularizer= regularizers.L2(l2 = 0.0001), activation='linear'))
  model.add(tf.keras.layers.Activation(activation = 'softmax'))
  model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
  model.save("model.h5")
  X_train, Y_train, x_validation_data, y_validation_data = split_data(X_train, Y_train, split_range=[0.8, 1.0])
  history = model.fit(X_train,Y_train,epochs = epochs,validation_data=(x_validation_data, y_validation_data),batch_size = batch_size)
  y_pred = model.predict(X_test)
  y_pred = np.argmax(y_pred,axis = 1)
  Y_test = np.argmax(Y_test,axis = 1)
  cm = confusion_matrix(Y_test,y_pred)
  return model, history, cm, y_pred