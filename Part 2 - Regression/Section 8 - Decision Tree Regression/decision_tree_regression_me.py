# Decision Tree Regression Model is not a good model in two variables.
import numpy as np # Non Linear and Non Continuos Regression Model
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

#Fitting Decision tree Model to the dataset 
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

#Prdicting a new result
y_pred=regressor.predict(6.5)

"""#Visualising Decision Tree results
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()""" #As non continous and non Linear so
# it will give us fault.lecture 66.

#Visualising Regression results(for higher resolution and smoother curve)
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show() # Spliting is clearly seen
# In each of the splitting value is the mean (constant) as told in
# the intuition lecture.