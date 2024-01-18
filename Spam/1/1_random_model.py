import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the dataset
df = pd.read_csv('spam.tsv', sep="\t")
# df = pd.read_csv('spam.csv')
# df = pd.read_csv('spam.csv', encoding='latin-1')

# Filter data based on labels
hamDf = df[df['label'] == "ham"]
spamDf = df[df['label'] == "spam"]

# hamDf = df[df['v1'] == "ham"]
# spamDf = df[df['v1'] == "spam"]

# Display filtered data

# hamDf=hamDf.sample(spamDf.shape[0])

finalDf=hamDf._append(spamDf,ignore_index=True)
X_train, X_test, Y_train, Y_test = train_test_split(finalDf['message'],finalDf['label'], test_size = 0.3, random_state = 1, shuffle = True,stratify = finalDf['label'])
# X_train, X_test, Y_train, Y_test = train_test_split(finalDf['v2'],finalDf['v1'], test_size = 0.3, random_state = 1, shuffle = True,stratify = finalDf['v1'])


# PipeLine
model= Pipeline ([('tfidf', TfidfVectorizer ()),('clf', RandomForestClassifier (n_estimators=100,n_jobs=-1))])



# model= Pipeline ([('tfidf', TfidfVectorizer ()),('model', SVC(C=1000,gamma='auto'))])

model.fit(X_train, Y_train)
Y_pred = model. predict(X_test)
print(accuracy_score(Y_test,Y_pred))
# joblib.dump(model,"mySVCModel1.pkl")
joblib.dump(model,"myModel.pkl")

# 0.9748803827751196