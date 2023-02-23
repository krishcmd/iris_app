from flask import Flask,render_template,request
from logging import FileHandler,WARNING

import pickle
import numpy as np

app=Flask(__name__)
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

model=pickle.load(open('model.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # making prediction
    return render_template('index.html', prediction_text='Entered features belong to the Species Classification : {}'.format(prediction[0]))
    

@app.errorhandler(500)
def internal_error(error):
    return "500 error"

    
if __name__=='__main__':
    app.run(port=8000)