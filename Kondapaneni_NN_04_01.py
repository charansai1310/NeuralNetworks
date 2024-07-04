import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import VGG19,VGG16

class CNN(object):
    def __init__(self):
        self.layer = []
        self.metric = "accuracy"
        self.optimizer = "SGD"
        self.lossfunction = "SparseCategoricalCrossentropy"
        self.model = None

    def add_input_layer(self, shape=(2,),name="" ):
        self.layer.append(Input(shape = shape,name = name))

    def append_dense_layer(self, num_nodes,activation="relu",name="",trainable=True):
        if self.model == None:
          x = self.layer[-1]
          dense = Dense(num_nodes,activation = activation,name = name,trainable = trainable)(x)
          self.layer.append(dense)
        else:
          dense = Dense(num_nodes,activation = activation,name = name,trainable = trainable)(self.model.layers[-1].output)
          model2 = Model(inputs = self.model.input, outputs = dense)
          self.model = model2

    def append_conv2d_layer(self, num_of_filters, kernel_size=3, padding='same', strides=1,activation="Relu",name="",trainable=True):
        if self.model == None:
          x = self.layer[-1]
          conv2d = Conv2D(num_of_filters,kernel_size = kernel_size,padding = padding,strides= strides,activation = activation,name = name,trainable = trainable)(x)
          self.layer.append(conv2d)
        else:
          conv2d = Conv2D(num_of_filters,kernel_size = kernel_size,padding = padding,strides= strides,activation = activation,name = name,trainable = trainable)(self.model.layers[-1].output)
          model2 = Model(inputs = self.model.input, outputs = conv2d)
          self.model = model2
        return conv2d

    def append_maxpooling2d_layer(self, pool_size=2, padding="same", strides=2,name=""):
        if self.model == None:
          x = self.layer[-1]
          maxpool = MaxPooling2D((2, 2),strides=(2, 2),name = name)(x)
          self.layer.append(maxpool)
        else:
          maxpool = MaxPooling2D((2, 2),strides=(2, 2),name = name)(self.model.layers[-1].output)
          model2 = Model(inputs = self.model.input, outputs = maxpool)
          self.model = model2
        return maxpool

    def append_flatten_layer(self,name=""):
        if self.model == None:
          x = self.layer[-1]
          flatten = Flatten(name = name)(x)
          self.layer.append(flatten)
        else:
          flatten = Flatten(name = name)(self.model.layers[-1].output)
          model2 = Model(inputs = self.model.input, outputs = flatten)
          self.model = model2
        return flatten

    def set_training_flag(self,layer_numbers=[],layer_names="",trainable_flag=True):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        for i in range(len(layer_numbers)):
          self.model.layers[layer_numbers[i]].trainable = trainable_flag
        self.model.layers.get_layer(layer_names).trainable = trainable_flag

    def get_weights_without_biases(self,layer_number=None,layer_name=""):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        y = None
        if layer_number == None:
          l = self.model.get_layer(layer_name)
          y = l.get_weights()
        else:
          y = self.model.layers[layer_number].get_weights()
        if y == None or len(y) == 0:
          return None 
        else:
          return np.array(y[0],dtype = "object")
        
    def get_biases(self,layer_number=None,layer_name=""):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        y = None
        if layer_number == None:
          l = self.model.get_layer(layer_name)
          y = l.get_weights()
        else:
          y = self.model.layers[layer_number].get_weights()
        if y == None or len(y) == 0:
          return None 
        else:
          return np.array(y[1],dtype = "object")

    def set_weights_without_biases(self,weights,layer_number=None,layer_name=""):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        if layer_number is not None:
          l = self.model.layers[layer_number]
          biases = l.get_weights()[1]
          weight_list = [weights,biases]
          l.set_weights(weight_list) 
        elif layer_name != "":
          l = self.model.get_layer(layer_name)
          biases = l.get_weights()[1]
          weight_list = [weights,biases]
          l.set_weights(weight_list)

    def set_biases(self,biases,layer_number=None,layer_name=""):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        if layer_number is not None:
          l = self.model.layers[layer_number]
          weights = l.get_weights()[0]
          weight_list = [weights,biases]
          l.set_weights(weight_list) 
        elif layer_name != "":
          l = self.model.get_layer(layer_name)
          weights = l.get_weights()[0]
          weight_list = [weights,biases]
          l.set_weights(weight_list)

    def remove_last_layer(self):
      x = self.odel.layers[-1]
      model2 = Model(inputs = self.model.input, outputs = self.model.layers[-2].output)
      self.model = model2
      return x

    def load_a_model(self,model_name="",model_file_name=""):
        if model_name != "":
          if model_name == "VGG19":
            self.model = VGG19()
          else:
            self.model = VGG16()
        else:
          self.model = load_model(model_file_name)
        return self.model

    def save_model(self,model_file_name=""):
        self.model.save(model_file_name)
        return self.model

    def set_loss_function(self, loss="SparseCategoricalCrossentropy"):
        self.lossfunction = loss

    def set_metric(self,metric):
        self.metric = metric

    def set_optimizer(self,optimizer="SGD",learning_rate=0.01,momentum=0.0):
        self.optimizer = optimizer

    def predict(self, X):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        y_pred = self.model.predict(X)
        return y_pred

    def evaluate(self,X,y):
        loss, metric = self.model.evaluate(X,y)
        return loss,metric 

    def train(self, X_train, y_train, batch_size, num_epochs):
        if self.model == None:
          self.model = Model(inputs = self.layer[0], outputs = self.layer[-1])
        self.model.compile(optimizer=self.optimizer, loss=self.lossfunction, metrics=self.metric)
        history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)
        return history.history['loss'],history.history['accuracy']

