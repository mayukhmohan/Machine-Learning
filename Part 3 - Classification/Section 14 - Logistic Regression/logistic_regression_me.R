dataset=read.csv('Social_Network_Ads.csv')
dataset=dataset[,3:5]

#Splitting the dataset into training and test set
#install.packages('caTools')

library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.75)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Feature SCaling Excluding categorical values
training_set[,1:2]=scale(training_set[,1:2])
test_set[,1:2]=scale(test_set[,1:2])

#Fitting Logistic Regression to the Training Set
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set) # Doubt binomial family

#Predicting the Test set results
prob_pred=predict(classifier, type = 'response', newdata = test_set[-3])
y_pred=ifelse(prob_pred > 0.5, 1, 0)

# Making the confusion metrics (Number correct and incorrect predictions)
cm = table(test_set[,3],y_pred)

#Visualising the Training set results
#install.packages('ElemStatLearn')
#Point is the  truth and region is the prediction
#Straight line is called the prediction boundary and it is a staright line because
#Logistic Regression is Linear Classifier (here 2D).
#(in a 3D it will be a straight plane).
library(ElemStatLearn)
set=training_set
X1=seq(min(set[,1])-1,max(set[,1])+1, by = 0.01)
X2=seq(min(set[,2])-1,max(set[,2])+1, by = 0.01)
grid_set=expand.grid(X1,X2)
colnames(grid_set)=c('Age','EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[,-3],
     main = 'Logistic Regression (Training Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add = TRUE)
points(grid_set,pch = '.',col = ifelse(y_grid==1,'springgreen3','tomato'))
points(set,pch = 21,bg=ifelse(set[,3]==1,'green4','red3'))


#Visualising the Test set results
set=test_set
X1=seq(min(set[,1])-1,max(set[,1])+1, by = 0.01) #Making the pixel values
X2=seq(min(set[,2])-1,max(set[,2])+1, by = 0.01) #Same as Python
grid_set=expand.grid(X1,X2) #Now making the pixel grid for real as well as imaginary users.
colnames(grid_set)=c('Age','EstimatedSalary') #Giving name to the columns of matrix 
prob_set = predict(classifier, type = 'response', newdata = grid_set) #Predicting for grid_set matrix for each skills
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[,-3],
     main = 'Logistic Regression (Test Set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1,X2,matrix(as.numeric(y_grid),length(X1),length(X2)),add = TRUE) 
points(grid_set,pch = '.',col = ifelse(y_grid==1,'springgreen3','tomato')) #For Backgroound Predictions
points(set,pch = 21,bg=ifelse(set[,3]==1,'green4','red3')) #For true points












