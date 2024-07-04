import numpy as np

def train_network(X_train,weight_list,layers):                      #code to train the network on train dataset
    output_list = []                                                #returns an Array of Outputs for the particular data point
    value = list(net_calculation(X_train,weight_list,layers))
    output_list = value
    return np.array(output_list)                                    #returns output values
        
def test_network(X_test,weight_list,layers):                        #code to test the network on test dataset
    output_list = []                                                #returns an Array of Outputs for the particular data point
    value = list(net_calculation(X_test,weight_list,layers))
    output_list = value
    return np.array(output_list)                                    #returns output values

def weight_adjustment(X_train,Y_train,weight_list,h,alpha,layers):
    #Finding the partial derivatives of the error w.r.to weights and updating them
    #first two loops update weight for 1st layer
    #second three loops update weight for 2nd layer
    #returns the updated weight list for the data point
    temp_weight_list = copy_list(weight_list)
    for j in range(layers[0]):
        for k in range(len(X_train)+1):
            temp_weight_list[0][j][k] = weight_list[0][j][k] + h                #adding weight 
            output_list = train_network(X_train, temp_weight_list, layers)      #finding output after changing weight  
            x1 = MSE(output_list,Y_train)                                       #finding mean squared error 
            temp_weight_list[0][j][k] = weight_list[0][j][k] - h                #subtracting weight
            output_list = train_network(X_train, temp_weight_list, layers)      #finding output after changing weight
            x2 = MSE(output_list,Y_train)                                       #finding mean squared error 
            w = (x1-x2)/(2*h)                                                   #calculating derivtive value
            temp_weight_list[0][j][k] = weight_list[0][j][k] 
            weight_list[0][j][k] = weight_list[0][j][k] - (alpha * w)           #updating new weights
    for i in range(1,len(layers)):
        for j in range(layers[i]):
            for k in range(layers[j]+1):
                temp_weight_list[i][j][k] = weight_list[i][j][k] + h            #adding weight
                output_list = train_network(X_train, temp_weight_list, layers)  #finding output after changing weight
                x1 = MSE(output_list,Y_train)                                   #finding mean squared error 
                temp_weight_list[i][j][k] = weight_list[i][j][k] - h            #subtracting weight
                output_list = train_network(X_train, temp_weight_list, layers)  #finding output after changing weight
                x2 = MSE(output_list,Y_train)                                   #finding mean squared error
                w = (x1-x2)/(2*h)                                               #calculating derivtive value
                temp_weight_list[i][j][k] = weight_list[i][j][k] 
                weight_list[i][j][k] = weight_list[i][j][k] - (alpha * w)       #updating new weights
    return weight_list                                                          #returns updated weight list 

def MSE(pred_value,true_value):                                                 #calculates mean squared error 
    return np.square(np.subtract(pred_value,true_value)).mean()

def sig(x):                                                                     #calculates sigmoid value
    return 1/(1 + np.exp(-x))

def copy_list(obj):                                                             #recursively copies an object and all nested object  
    if isinstance(obj, (list, tuple)):
        return type(obj)(copy_list(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: copy_list(value) for key, value in obj.items()}
    from copy import deepcopy as copy 
    if isinstance(obj, (list, tuple)):
        return type(obj)(copy_list(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: copy_list(value) for key, value in obj.items()}
    return copy(obj)

def net_calculation(data,weights,layers):                                       #calculating net values 
    for i in range(len(layers)):
        outputs = []
        for j in range(layers[i]):
            net = 0
            for k in range(len(data)):
                net = net + (weights[i][j][k+1]*data[k])
            net = net + weights[i][j][0]
            outputs.append(sig(net))
        data = outputs
    return data                                                                 #returns the calculated net value

def weight_initialization(rows,columns,seed):                                   #inititalising random weights for the weight matrix
    np.random.seed(seed)
    weight_matrix_layer = []
    for i in range(rows):
        weights = []
        for j in range(columns):
            weights.append(np.random.randn())
        weight_matrix_layer.append(weights)
    return np.array(weight_matrix_layer)                                        #returns the initialized weight matrix

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h,seed):
    X_train = np.transpose(X_train)
    Y_train = np.transpose(Y_train)                                             #transposing the input data
    X_test = np.transpose(X_test)
    Y_test = np.transpose(Y_test)
    n = len(X_train)
    m = len(X_test)
    weight_list = []
    mse_list = []
    output_list = []
    return_list = []
    outputs = []
    np.random.seed(seed)
    weight_list.append(weight_initialization(layers[0],len(X_train[0])+1,seed)) #Initialising Weights of First Layer
    for j in range(1,len(layers)):
        weight_list.append(weight_initialization(layers[j],layers[j-1]+1,seed)) #Initialising Weights of All Other Layers
    for i in range(epochs):                                                     #loop for epochs
        for j in range(n):                                                      #loop for every data in the train data 
            output_list = train_network(X_train[j],weight_list, layers)         #output calculation
            weight_list = weight_adjustment(X_train[j],Y_train[j],weight_list,h,alpha,layers) #weight adjustment for first epoch
        test_mse_list = []
        for j in range(m):                                                      #loop for every data in the test data                                                   
            output_list = test_network(X_test[j], weight_list, layers)          #output calculation
            test_mse_list.append(MSE(output_list,Y_test[j]))                    #calculates mean squared error for every data point
        result_mse = np.mean(test_mse_list)                                     
        mse_list.append(result_mse)                                             #calculates mean squared error the epoch
    outputs = []
    for j in range(m):                                                          #code to return output when test data used
        outputs.append(test_network(X_test[j], weight_list, layers))
    return_list.append(weight_list)                                             #appending weight matrices to return list
    return_list.append(mse_list)                                                #appending mean squred error values to the return list
    outputs = np.array(outputs).T                                               #transposing the output values
    return_list.append(outputs)                                                 #appending outputs to the return list     
    return return_list                                                          #return list with weight matrices, error values and outputs