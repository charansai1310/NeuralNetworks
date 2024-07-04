import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def linear(net):                                                                                    #Activation Functions
  return net

def sigmoid(net):                                                                                   #Activation Functions
  return 1/(1+tf.exp(-net))

def relu(net):                                                                                      #Activation Functions
  return tf.maximum(0.,net)

def weight_initialization(rows,columns,seed):                                                       #Weight Initializations
  np.random.seed(seed)
  w = np.empty([rows,columns],dtype = np.float32)
  for i in range(rows) :
    for j in range(columns) :
      w[i,j] = np.random.randn()
  return w

def svm(y_pred,y):                                                                                  #SVM Loss Function
  return tf.reduce_mean(tf.maximum(0.0,1-y_pred*y))

def cross_entropy_loss(y_pred, y):                                                                  #Cross Entropy Loss Function
  ce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
  return tf.reduce_mean(ce)

def mse(y_pred,y):                                                                                  #MSE Loss Function
  return tf.reduce_mean(tf.reduce_mean(tf.square(y - y_pred)))

def train_model(x_data,layers,activations,w):                                                       #Model Training 
  for i in range(len(w)):
    z = tf.matmul(x_data,w[i])
    if activations[i] == "relu":
      x_data = relu(z)
    elif activations[i] == "linear":
      x_data = linear(z)
    else:
      x_data = sigmoid(z)
    if len(w)-i != 1:
      bias_array = tf.ones([x_data.shape[0],1])
      x_data = tf.concat([bias_array,x_data], 1)
  return x_data

def train_model_(x_data,layers,activations,w):                                                      #Model Training Before Epochs               
  for i in range(len(w)):
    z = tf.matmul(x_data,w[i])
    if activations[i] == "relu":
      x_data = relu(z)
    elif activations[i] == "linear":
      x_data = linear(z)
    else:
      x_data = sigmoid(z)
    if len(w)-i != 1:
      bias_array = tf.ones([x_data.shape[0],1])
      x_data = tf.concat([bias_array,x_data], 1)
  return x_data

def train_network(x_data,y_data,layers,activations,w,alpha,loss):                                 #Network Training  
  with tf.GradientTape() as tape:
    y_pred = train_model(x_data,layers,activations,w)
    if loss == 'svm':
      batch_loss = svm(y_pred, y_data)
    elif loss == 'mse':
      batch_loss = mse(y_pred, y_data)
    else:
      batch_loss = cross_entropy_loss(y_pred, y_data)
  grads = tape.gradient(batch_loss,w)                                                             #Gradient Calculation
  weight = []
  for i in range(len(w)):
    k = w[i] - alpha * grads[i]                                                                   #Weight Updation
    weight.append(tf.Variable(k))
  return weight                                                                                   #Returns Weight

def train_network_before_epoch(x_data,y_data,layers,activations,w):                               #Network Training Before Epochs 
  y_pred = train_model_(x_data,layers,activations,w)
  return y_pred

def val_network(x_data,y_data,layers,activations,w,loss):                                         #Validation Network 
  y_pred = train_model(x_data,layers,activations,w)                                               #Values Prediction
  if loss == 'svm':                                                                             
    batch_loss = svm(y_pred, y_data)
  elif loss == 'mse':
    batch_loss = mse(y_pred, y_data)
  else:
    batch_loss = cross_entropy_loss(y_pred, y_data)
  return batch_loss,y_pred                                                                        #Returning Loss And Predicted Values  

def generate_batches(X, y, batch_size=32):                                                        #Batches Generation
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs,loss="svm",validation_split=[0.8,1.0],weights=None,seed=2):              #MLP Implementation
  return_list = []
  err=[]
  outputs = []
  w = []
  start = int(X_train.shape[0] * validation_split[0])
  stop = int(X_train.shape[0] * validation_split[1])
  x_validation_data = X_train[start:stop]
  x_train_data = np.delete(X_train,np.s_[start:stop],axis = 0)
  y_validation_data = Y_train[start:stop]
  y_train_data = np.delete(Y_train,np.s_[start:stop],axis = 0)
  n = len(layers)
  out = np.empty((y_validation_data.shape[0],layers[n-1]),dtype = np.float32)
  if weights == None:                                                                                          #Random Weights Initializations
    w.append(tf.Variable(weight_initialization(x_train_data.shape[1]+1,layers[0],seed),trainable= True))
    for i in range(1,len(layers)):
      w.append(tf.Variable(weight_initialization(layers[i-1]+1,layers[i],seed),trainable= True))
  else:
    for i in weights:
      w.append(tf.Variable(i,trainable = True))                                                                 #Given Weights Initializations
  bias_array = tf.ones([x_validation_data.shape[0],1])
  x_validation_data = tf.concat([bias_array,x_validation_data], 1)
  out = train_network_before_epoch(x_validation_data,y_validation_data,layers,activations,w)                    #Training Model before Epochs
  x_validation_data = X_train[start:stop]
  y_validation_data = Y_train[start:stop]
  for epoch in range(epochs):                                                                                   #Loop for Epochs
    for x_batch,y_batch in generate_batches(x_train_data,y_train_data,batch_size):                              #Loop for Batch Training
      bias_array = tf.ones([x_batch.shape[0],1])
      x_batch = tf.concat([bias_array,x_batch], 1)
      w = train_network(x_batch,y_batch,layers,activations,w,alpha,loss)

    x_validation_data = X_train[start:stop]
    y_validation_data = Y_train[start:stop]
    bias_array = tf.ones([x_validation_data.shape[0],1])
    x_validation_data = tf.concat([bias_array,x_validation_data], 1)
    error , out = val_network(x_validation_data,y_validation_data,layers,activations,w,loss)                    #Validation Step
    err.append(error)
  return_list.append(w)                                                                                         #Weight List
  return_list.append(err)                                                                                       #Error List
  return_list.append(out)                                                                                       #Outputt Array
  return return_list