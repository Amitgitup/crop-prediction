from flask import Flask, jsonify, request, render_template
import numpy as np
import pickle

app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    feature = [np.array(float_feature)]
    predictions = model.predict(feature)
    return render_template('index.html', predicted_text="The Predicted crop is {}".format(predictions))

if __name__=="__main__":
    app.run(debug=True)