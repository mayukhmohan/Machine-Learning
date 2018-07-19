import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)#Ignoring double quotes

"""import nltk
nltk.download('popular')"""
#Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', df['Review'][i])
    #We want to keep the letters a-z and A-Z and replacing the abandoned letters with space.
    review = review.lower()
    review = review.split()
    #Stemming to avoid sparsity in matrix such as loved -> love
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)

# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500) # To reduce sparsity we can reduce max_sparsity or reduce dimensions
X=cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting Naive Bayes to the Traning set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting The test set results
y_pred=classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 





