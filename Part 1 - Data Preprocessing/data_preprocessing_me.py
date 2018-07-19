import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('Data.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#we replace the missing data by the column's mean

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])

#But LabelEncoder assigns 0,1,2 etc so that
#machine learning algos can think there maybe priorities
#which is actually not of course.

onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Splitting the dataset into training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling, as there  are many variabbles with different 
#ranges, so one variable may be dominated by the others.
#To keep thing unbiased for all the variables feature scaling s done.
#makes very fast algo, (eucledian distance based),Machine Learning model converges fast

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test) 
#depend on context dummy are scaled or not
#do not need feature scaling for dependent variable 
#as far as that is not of so large range



















