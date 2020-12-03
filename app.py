import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import CountVectorizer
import pickle

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

import pandas as pd
df = pd.read_csv('data/amazon_imdb_yelp_labelled.txt', names=['review', 'sentiment'], sep='\t')
from sklearn.model_selection import train_test_split
reviews = df['review'].values
labels = df['sentiment'].values
reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.2, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit(reviews_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print("int_features", int_features)
    final_features = vectorizer.transform(int_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 0:
        sentiment = "Negative"
    elif output == 1:
        sentiment = "Positive"

    return render_template('index.html', prediction_text='sentiment is {}'.format(sentiment))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)