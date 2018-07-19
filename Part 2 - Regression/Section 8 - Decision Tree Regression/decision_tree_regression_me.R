dataset=read.csv('Position_Salaries.csv')
dataset=dataset[2:3]

#Splitting the dataset into training and test set
#install.packages('caTools')
# 
# library(caTools)
# set.seed(123)
# split=sample.split(dataset$Purchased,SplitRatio = 0.8)
# training_set=subset(dataset,split==TRUE)
# test_set=subset(dataset,split==FALSE)

#Feature SCaling Excluding categorical values
# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])


#Fitting Decision Tree Regression Model to the dataset
# install.packages('rpart')
library(rpart)
regressor=rpart(formula = Salary ~ .,
                data = dataset,
                control = rpart.control(minsplit = 1))
#Normally split then whole thing is in one split.
#So the constant line comes.


#Predicting a new result with Decision Tree Regression Model
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))


# #Visualising the Decision Tree Regression Model results
# library(ggplot2)
# ggplot() +
#   geom_point(aes(x =dataset$Level , y =dataset$Salary),
#              colour = 'red') +
#   geom_line(aes(x =dataset$Level , y =predict(regressor, newdata = dataset)),
#             colour = 'blue') +
#   ggtitle('Truth or Bluff (Decision Tree Regression Model)') +
#   xlab('Level')+
#   ylab('Salary')


#Visualising the Decision Tree Regression Model results (Creating Smooth curve and increase the resolution)
library(ggplot2)
x_grid=seq(min(dataset$Level), max(dataset$Level),0.01)
ggplot() +
  geom_point(aes(x =dataset$Level , y =dataset$Salary),
             colour = 'red') +
  geom_line(aes(x =x_grid , y =predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression Model)') +
  xlab('Level')+
  ylab('Salary')

