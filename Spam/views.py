from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib
import numpy as np
from collections import Counter

# https://github.com/Faisalhuss/email-spam-detection-using-ML

loaded_tfidf2 = joblib.load(os.path.dirname(__file__) + "\\2\\tfidf_vectorizer.pkl")
loaded_model2 = joblib.load(os.path.dirname(__file__) + "\\2\\best_svc_model.pkl")


model1 = joblib.load(os.path.dirname(__file__) + "\\1\\myModel.pkl")

loaded_vectorizer5 = joblib.load(os.path.dirname(__file__) + "\\5\\tfidf_vectorizer2.pkl")
loaded_model5 = joblib.load(os.path.dirname(__file__) + "\\5\\adaboost_model_tuned2.pkl")

loaded_model = joblib.load(os.path.dirname(__file__) + '\\4\\best_model_nb.pkl')
loaded_tfidf = joblib.load(os.path.dirname(__file__) + '\\4\\tfidf_vectorizer_nb.pkl')

loaded_model1 = joblib.load(os.path.dirname(__file__) + '\\3\\spam_classifier_model.pkl')
loaded_tfidf1 = joblib.load(os.path.dirname(__file__) + '\\3\\tfidf_vectorizer-nb.pkl')


def index(request):
    return render(request, 'index.html')

def checkSpam(request):
    if request.method == "POST":
        algo = request.POST.get("algo")
        rawData = request.POST.get('rawdata')
        param = {}
        if not rawData.strip():
            param = {"answer": "No input"}

        elif algo == "Algo-1":
            finalAns = model1.predict([rawData])[0]
            param = {"answer": finalAns}
    
        elif algo == "Algo-2":
            input_vectorized = loaded_tfidf2.transform([rawData])
            result = loaded_model2.predict(input_vectorized)[0]
            param = {"answer": result}

        elif algo == "Algo-3":      
            result = loaded_model1.predict([rawData])[0]
            param = {"answer": result}
  
        

        elif algo == "Algo-4":
            text_vectorized = loaded_tfidf.transform([rawData])
            prediction = loaded_model.predict(text_vectorized)[0]
            param = {"answer": prediction}

        elif algo == "Algo-5":
            vector_input = loaded_vectorizer5.transform([rawData])
            result = loaded_model5.predict(vector_input)[0]
            param = {"answer": result}
        else:
            param = {"answer": "No algorithm selected"}
        return render(request, 'output2.html', param)
    else:
        return render(request, 'index.html')




def predictEmailType(request):
    if request.method == "POST":
        rawData = request.POST.get('rawdata')
        param = {}
        if not rawData.strip():
            param = {"answer": "No input", "prediction": ""}
        else:
            # Predict with all five algorithms
            algo_names = ['Random Forest', 'SVM', 'Naive Bayes', 'Bagging', 'AdaBoost']
            algo_predictions = []
            algo_predictions.append(model1.predict([rawData])[0].capitalize())
            algo_predictions.append(loaded_model2.predict(loaded_tfidf2.transform([rawData]))[0].capitalize())
            algo_predictions.append(loaded_model1.predict([rawData])[0].capitalize())
            algo_predictions.append(loaded_model.predict(loaded_tfidf.transform([rawData]))[0].capitalize())
            algo_predictions.append(loaded_model5.predict(loaded_vectorizer5.transform([rawData]))[0].capitalize())

            # Prepare detailed answer
            detailed_answer = ""
            for name, prediction in zip(algo_names, algo_predictions):
                detailed_answer += f"- {name} predicted: {prediction}<br>"

            # Count the predictions
            prediction_counts = Counter(algo_predictions)

            # Find the most common prediction
            final_prediction = prediction_counts.most_common(1)[0][0]

            detailed_answer += f"<br>Final prediction (based on majority vote): {final_prediction}"

            param = {"answer": detailed_answer, "prediction": final_prediction}

        return render(request, 'output5.html', param)
    else:
        return render(request, 'index.html')

