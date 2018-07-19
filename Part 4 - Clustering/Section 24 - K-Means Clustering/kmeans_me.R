dataset = read.csv('Mall_Customers.csv')
X=dataset[4:5]

# Elbow Method
set.seed(6)
wcss=vector()
for (i in 1:10) wcss[i] <- sum(kmeans(X,i)$withinss)
plot(1:10,wcss,type='b',main = 'Clusters of Clients',xlab = 'Number of Clusters',ylab = 'WCSS')

#Applying K-Means to the mall dataset
set.seed(29)
kmeans=kmeans(X,5,iter.max = 300,nstart = 10)

# Visualising the Clusters
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = 'Cluster of Lines',
         xlab = 'Annual income',
         ylab = 'Spending Score')