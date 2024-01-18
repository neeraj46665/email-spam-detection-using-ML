import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the TSV file
data = pd.read_csv('spam.tsv', sep='\t')

# Assuming your columns are named 'label' and 'message'
X = data['message']
Y = data['label']

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X_tfidf, Y)

# Split the resampled data
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)

# Initialize and train the AdaBoost Classifier
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_model.fit(X_train, Y_train)

# Make predictions on the test set
Y_pred = adaboost_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, pos_label='spam')
recall = recall_score(Y_test, Y_pred, pos_label='spam')
f1 = f1_score(Y_test, Y_pred, pos_label='spam')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Save the model and vectorizer to .pkl files
joblib.dump(adaboost_model, 'adaboost_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
