dataset_original = read.delim('Restaurant_Reviews.tsv',quote = '',stringsAsFactors = FALSE)

# Cleaning the Texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset$Review))
corpus = tm_map(corpus,content_transformer(tolower))
corpus = tm_map(corpus,removeNumbers)
corpus = tm_map(corpus,removePunctuation)
corpus = tm_map(corpus,removeWords,stopwords())
corpus = tm_map(corpus,stemDocument)
corpus = tm_map(corpus,stripWhitespace)

#Creating the Bag of Words Model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm,0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

#Random Forest Classification
#Encoding the target feature as factor
dataset$Liked=factor(dataset$Liked,levels = c(0,1))

#Splitting the dataset into training and test set
library(caTools)
set.seed(123)
split=sample.split(dataset$Liked,SplitRatio = 0.80)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#Fitting Random Forest Classification to the Training Set
#install.packages('randomForest')
library(randomForest)
classifier=randomForest(x = training_set[-692],
                        y = training_set$Liked,
                        ntree = 10)

#Predicting the Test set results
y_pred=predict(classifier, newdata = test_set[-692])

# Making the confusion metrics
cm = table(test_set[,692],y_pred)
