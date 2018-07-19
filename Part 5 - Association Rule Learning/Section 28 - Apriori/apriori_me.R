dataset=read.csv('Market_Basket_Optimisation.csv',header = FALSE)
# install.packages('arules')
library(arules)
dataset=read.transactions('Market_Basket_Optimisation.csv',sep = ',',rm.duplicates = TRUE)
#Sparsing matrix the data

#info visualising and details
summary(dataset)
itemFrequencyPlot(dataset,topN=10)

# Training Apriori on the dataset
rules = apriori(data = dataset,parameter = list(support = 0.004,confidence = 0.2))
# atleast purchased 3 times a day 3*7/7500 minimum support, default confidence 0.8.
# then keep decreasing to smaller confidence.

#Visualising the results
inspect(sort(rules,by = 'lift')[1:10])

