import numpy as np
import pickle
from flask import Flask,request,render_template 

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if prediction == 1:
        output ="Positive"
    else:
        output = "Negative"

    return render_template('index.html',prediction_text="The patient is showing {} results for diabetics".format(output))


if __name__ == "__main__":
    app.run(debug=True)