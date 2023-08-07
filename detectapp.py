from flask import Flask, render_template,request
import pickle
import pandas as pd
import sklearn
import re
import numpy as np
import string
LR=pickle.load(open('newsmodel.pkl','rb'))
vector=pickle.load(open('vector.pkl','rb'))
app= Flask(__name__)
@app.route('/')
def index():
    return render_template("index2.html")

@app.route('/predict',methods=['POST'])


def process1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
def manual(news):
    test_news = {"text": [news]}
    new_def_test = pd.DataFrame(test_news)
    new_def_test["text"] = process1(new_def_test["text"])
    new_x_test = new_def_test["text"]
    new_xv_test = vector.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    if pred_LR[0] == 1:
        result = "News is FAKE"
    else:
        result = "News is REAL"
    return render_template('index2.html', result=result)

def predict_placement():
    news = request.form.get('news')
    manual(news)

if __name__=='__main__':
    app.run(debug=True)