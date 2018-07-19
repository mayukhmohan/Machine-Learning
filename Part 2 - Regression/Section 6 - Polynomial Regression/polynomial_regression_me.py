import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('Position_Salaries.csv')
X=df.iloc[:,1:2].values # X is a matrix
y=df.iloc[:,-1].values # y is a vector

#Splitting the dataset into training and test set (Here we dont have many observation)
# and we have to make very accurate prediction. So in this case we are taking the whole dataset to train the model. 
"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)"""

#Feature Scaling, 
"""from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test) """

#Fitting Linear Regression to the dataset(for comparing) 
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising Linear Regression
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualising Polynomial Regression
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Prdicting anew result with linear Regression
lin_reg.predict(6.5)

#Prdicting anew result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
