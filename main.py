import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np
from scipy import misc

app = Flask(__name__)

#main page routing
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

#data prediction
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        #upload file
        file = request.files['image']
        if not file:
            return render_template('index.html', label="No Files")
        
        #read image
        img = misc.imread(file)
        img = img[:, :, :3]
        img = img.reshape(1, -1)

        #predict image
        prediction = model.predict(img)
        label = str(np.squeeze(prediction))

        if label == '10':
            label = '0'
        
        return render_template('index.html', label=label)

if __name__ == '__main__':
    #load model
    # ml/model.py 선 실행 후 생성
    model = joblib.load('/model.pkl')
    #run flask service
    app.run(host='0.0.0.0', port=8000, debug=True)