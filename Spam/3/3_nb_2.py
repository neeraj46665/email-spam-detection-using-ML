import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize 
import joblib

ps = PorterStemmer()

# Load your dataset
data = pd.read_csv('spam.tsv', sep='\t')

# Assuming your columns are named 'label' and 'message'
X = data['message']
Y = data['label']

# Text preprocessing function (replace this with your actual preprocessing steps)
def preprocess_text(text):
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

# Preprocess the text
processed_text = X.apply(preprocess_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(processed_text)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.3, random_state=42)

# Initialize and train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)

# Save the model and TF-IDF vectorizer
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer-nb.pkl')

# Predictions on test set
Y_pred = nb_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Printing the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
