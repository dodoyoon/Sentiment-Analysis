import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('./data/amazon_imdb_yelp_labelled.txt', names=['review', 'sentiment'], sep='\t')

reviews = df['review'].values
labels = df['sentiment'].values
reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=1000)

punctuations = string.punctuation
parser = English()
stopwords = list(STOP_WORDS)

vectorizer = CountVectorizer()
vectorizer.fit(reviews_train)

X_train = vectorizer.transform(reviews_train)
X_test = vectorizer.transform(reviews_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
# print(X_test)
# print(y_test)
print("Accuracy:", accuracy)

pickle.dump(classifier, open('model/model.pkl', 'wb'))