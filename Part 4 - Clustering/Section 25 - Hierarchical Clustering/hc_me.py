import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
df=pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,[3,4]].values

# plotting Dendogram to find a optimal number of cluster
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
#here we are minimizing within cluster variance.Instead of minimizing sum of squares.
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distances')
plt.show()

# Fiiting Hierarchial Clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1], s=100,c='red',label='Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1], s=100,c='blue',label='Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1], s=100,c='green',label='Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1], s=100,c='cyan',label='Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1], s=100,c='magenta',label='Sensible')
plt.title('Cluster of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

