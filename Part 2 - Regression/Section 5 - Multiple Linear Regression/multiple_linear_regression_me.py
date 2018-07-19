import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('50_Startups.csv')
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
X=X[:,1:] #Libraries may take care about the fact many times

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Feature Scaling, 
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test) """

#Fitting Multiple Linear Regression to the Traning Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test set results
y_pred=regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm #statmodels will not consider the equation
#as y=b0+b1x1+b2x2+... rather it will consider y=b1x1+b2b2+b3x3+..
#so we have to take x0=1 for all cases so that y=b0x0+b1x1+b2x2+....
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)
X_opt= X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit() #ordinary least square
regressor_ols.summary() #tells us about the p value and so many things it shows
#highest value is 0.99 way above the significant level 0.05
#so remove the variable x2. Removing must be doe after seeing the X not X_opt
X_opt= X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_ols.summary()
X_opt= X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_ols.summary()
X_opt= X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_ols.summary()
X_opt= X[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit() 
regressor_ols.summary()







