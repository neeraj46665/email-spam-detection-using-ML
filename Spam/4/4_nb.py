import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load your dataset
data = pd.read_csv('spam.tsv', sep='\t')

# Assuming your columns are named 'label' and 'message'
X = data['message']
Y = data['label']

# Text preprocessing function (replace this with your actual preprocessing steps)
def preprocess_text(text):
    return text.lower()  # Placeholder for actual preprocessing

# Preprocess the text
processed_text = X.apply(preprocess_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(processed_text)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.3, random_state=42)

# Initialize the Multinomial Naive Bayes model
nb_model = MultinomialNB()

# Define the parameter grid to search
param_grid = {'alpha': [0.1, 0.5, 1.0]}

# Perform Grid Search to find the best hyperparameters
grid_search = GridSearchCV(nb_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, Y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

joblib.dump(best_model, 'best_model_nb.pkl')

# # Save the TF-IDF vectorizer to a .pkl file
joblib.dump(tfidf, 'tfidf_vectorizer_nb.pkl')

# Use the best model for predictions
Y_pred = best_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Print the metrics and best hyperparameters
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Best hyperparameters: {best_params}")


# Accuracy: 0.9880382775119617
# Precision: 0.987971054608723
# Recall: 0.9880382775119617
# F1-score: 0.9879926453429176
# Best hyperparameters: {'alpha': 0.1}
