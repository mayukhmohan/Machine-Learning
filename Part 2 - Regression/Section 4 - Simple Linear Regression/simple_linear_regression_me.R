dataset=read.csv('Salary_Data.csv')

#Splitting the dataset into training and test set
#install.packages('caTools')

library(caTools)
set.seed(123)
split=sample.split(dataset$Salary,SplitRatio = 2/3)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Feature SCaling Excluding categorical values
# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])

#Fitting Simple Linear Regression to the training set
regressor=lm(formula = Salary ~ YearsExperience, data = training_set)
#summary(regressor) gives us many things
#three star means highly significant and correlated
#data and less than 5% p value also signifies that.

#Prdicting the test set result
y_pred=predict(regressor,newdata = test_set)

#Visualising the observations
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
            color='blue') +
  ggtitle('Salary vs Experience (Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

ggplot() +
  geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary),
             color = 'red') +
  geom_line(aes(x = training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
            color='blue') +
  ggtitle('Salary vs Experience (Test set)') +
  xlab('Years of Experience') +
  ylab('Salary')






