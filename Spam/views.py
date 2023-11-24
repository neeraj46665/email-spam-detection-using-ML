from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = joblib.load(os.path.dirname(__file__) + "\\vectorizer.pkl",'rb')
model = joblib.load(os.path.dirname(__file__) + "\\model.pkl",'rb')
model1 = joblib.load(os.path.dirname(__file__) + "\\mySVCModel.pkl")
model2 = joblib.load(os.path.dirname(__file__) + "\\myModel.pkl")

def index(request):
    return render(request, 'index.html')

def checkSpam(request):
    if request.method == "POST":
        algo = request.POST.get("algo")
        rawData = request.POST.get('rawdata')
        param = {}

        if algo == "Algo-1":
            finalAns = model1.predict([rawData])[0]
            param = {"answer": finalAns}
        
        elif algo == "Algo-2":
            finalAns = model2.predict([rawData])[0]
            param = {"answer": finalAns}

        elif algo == "Algo-3":
            # Here, it seems like you intended to use `rawData` instead of `finalAns`
            
            transformed_sms = transform_text(rawData) 
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            finalAns = "ham" if result == 0 else "Spam"
            param = {"answer": finalAns}
            

        return render(request, 'output.html', param)
    else:
        return render(request, 'index.html')