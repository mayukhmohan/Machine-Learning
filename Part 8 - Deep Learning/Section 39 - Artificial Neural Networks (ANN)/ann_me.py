import tensorflow
import theano
import keras

# Part 1 Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] # Avoiding Dummy Variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 Making the ANN
# Importing the keras libraries and packages
from keras.models import Sequential 
from keras.layers import Dense

#Initializing the ANN (this model is classifier)
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(input_dim = 11,units = 6,kernel_initializer = 'uniform',activation = 'relu'))
# Number of hidden layers nodes is avg of the number of input nodes and output nodes (trick/tip)
# Number of hidden layers nodes are 6, weights are uniformly distributed
# and close to zero.Activatiion function for hidden layers is rectified -> relu.
# input_dim is required for first hidden layers as it does not know the number
# of independent variables as input nodes. 

#Second Hidden Layer
classifier.add(Dense(units = 6,kernel_initializer = 'uniform',activation = 'relu'))
# As it know beforehand its Input, so no input_dim

# Adding the output layer
classifier.add(Dense(units = 1,kernel_initializer = 'uniform',activation = 'sigmoid'))
# Dependent Variable for more than two we have to use softmax function.

# Compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
# Adam is a stochastic gradient descent algorithm
# To calculate the loss (binary outcome) we have to use binary _crossentropy
# Fortwo or more than two dependent variable we have to use category_crossentropy
# accuracy will improve on each epoch(as it is metric)

# Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size = 10,epochs = 100)

# Part 3 Making Predictions and Evaluating the model
# Predicting The test set results
y_pred=classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 







