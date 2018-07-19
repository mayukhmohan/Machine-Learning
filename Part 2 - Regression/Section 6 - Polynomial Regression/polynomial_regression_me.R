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

#Fitting Linear Regression to the dataset
lin_reg=lm(formula = Salary ~ .,
           data = dataset)

#Fitting Linear Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg=lm(formula = Salary ~ .,
            data = dataset)

#Visualising the Linear Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x =dataset$Level , y =dataset$Salary),
             colour = 'red') +
  geom_line(aes(x =dataset$Level , y =predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level')+
  ylab('Salary')


#Visualising the Polynomials Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x =dataset$Level , y =dataset$Salary),
             colour = 'red') +
  geom_line(aes(x =dataset$Level , y =predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level')+
  ylab('Salary')

#Predicting a new result with Linear regression
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))


#Predicting a new result with Polynomial regression
y_pred_dash = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3= 6.5^3,
                                                Level4=6.5^4))
