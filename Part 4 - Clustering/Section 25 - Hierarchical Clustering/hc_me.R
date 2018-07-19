dataset = read.csv('Mall_Customers.csv')
X=dataset[4:5]

# Using the dendrogram
dendrogram = hclust(dist(X,method = 'euclidean'),method = 'ward.D')
plot(dendrogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean Distances')

# Fitting Hierarchial Clustering to the mall dataset
hc = hclust(dist(X,method = 'euclidean'),method = 'ward.D')
y_hc=cutree(hc,5)

# VIsualising the clusters
library(cluster)
clusplot(X,
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = 'Cluster of Lines',
         xlab = 'Annual income',
         ylab = 'Spending Score')