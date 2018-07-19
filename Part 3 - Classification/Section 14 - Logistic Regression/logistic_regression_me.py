# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression tothe Traning set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predicting The test set results
y_pred=classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #Upper Left Down Right shows correct
# Upper Right Down Left shows the wrong.

#Visualising the Training set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                     np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01)) 
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label=j)
plt.title('Logistic Regression (Traning Set)')
plt.xlabel('Age')
plt.ylabel('Estimatted Salary')
plt.legend()
plt.show()

#Visualising the Test set results 
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test #Just creating local variables.
X1, X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),
                     np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01)) 
# It is actually specifying the pixel values with 0.01 gap.Now we dont want to squeeze the
#points to the corners so we take min-1 and max+1
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha=0.75, cmap=ListedColormap(('red','green')))
# Applying the classifiers on all the pixel classifiers observation points colurs 
# all the red and green pixel points. We use contour function to make the contour line.
# then we use predict function to predict each of the pixel point is in class0 or class 1
# and colouring them accordingly.
plt.xlim(X1.min(),X1.max()) # putting x limits
plt.ylim(X2.min(),X2.max()) # putting y limits
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label=j)
# plotting all the points in scatter plot.    
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimatted Salary')
plt.legend()
plt.show()




















