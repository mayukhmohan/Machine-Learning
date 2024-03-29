dataset=read.csv('Social_Network_Ads.csv')
dataset=dataset[,3:5]

#Splitting the dataset into training and test set
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Feature SCaling Excluding categorical values
training_set[,1:2]=scale(training_set[,1:2])
test_set[,1:2]=scale(test_set[,1:2])

#Fitting Classifier to the Training Set
#Create your Classifier here 

#Predicting the Test set results
y_pred=predict(classifier, type = 'response', newdata = test_set[-3])

# Making the confusion metrics
cm = table(test_set[,3],y_pred)

#Visualising the Training set results
library(ElemStatLearn)
set=training_set
X1=seq(min(set[,1])-1,max(set[,1])+1, by = 0.01)
X2=seq(min(set[,2])-1,max(set[,2])+1, by = 0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')
y_grid = predict(classifier, type = 'response', newdata = grid_set)
plot(set[,-3],
     main = 'Classifier (Training Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add = TRUE)
points(grid_set,pch = '.',col = ifelse(y_grid==1,'springgreen3','tomato'))
points(set,pch = 21,bg=ifelse(set[,3]==1,'green4','red3'))


#Visualising the Test set results
set=test_set
X1=seq(min(set[,1])-1,max(set[,1])+1, by = 0.01) 
X2=seq(min(set[,2])-1,max(set[,2])+1, by = 0.01) 
grid_set=expand.grid(X1,X2) 
colnames(grid_set)=c('Age','EstimatedSalary')  
y_grid = predict(classifier, type = 'response', newdata = grid_set) 
plot(set[,-3],
     main = 'Classifier (Test Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add = TRUE) 
points(grid_set,pch = '.',col = ifelse(y_grid==1,'springgreen3','tomato')) 
points(set,pch = 21,bg=ifelse(set[,3]==1,'green4','red3')) 

