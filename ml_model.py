import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import nltk
from nltk.corpus import stopwords

# Downloading stopwords if not already done
nltk.download("stopwords")

# Preprocessing function to clean the data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Loading the dataset
data = pd.read_csv(r'C:\Users\suraj\Desktop\spam detect\spam.csv', encoding="latin-1")

# Inspect the columns to see what's actually there
print(data.columns)

# Show the first few rows to check the data
print(data.head(10))

# Now, drop the columns that are not needed (based on the actual column names)
# This is where we use the actual column names after inspecting them
data = data[['Category', 'Message']]  # Assuming these are the correct columns based on inspection

# Convert category labels to binary (ham: 0, spam: 1)
data['Category'] = data['Category'].map({'ham': 0, 'spam': 1})

# Apply preprocessing to the messages
data['Message'] = data['Message'].apply(preprocess_text)

# Define feature (X) and target (y)
X = data['Message']
y = data['Category']

# Check for missing values
print(data.isnull().sum())

# Convert text data to numerical data using CountVectorizer
cv = CountVectorizer()
X_cv = cv.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model's performance
print(f"Training Accuracy: {model.score(X_train, y_train)}")
print(f"Testing Accuracy: {model.score(X_test, y_test)}")

# Save the trained model and vectorizer
model_path = r'C:\Users\suraj\Desktop\spam detect\spam123.pkl'
vectorizer_path = r'C:\Users\suraj\Desktop\spam detect\vec123.pkl'

# Save the trained model
with open(model_path, 'wb') as model_file:
    pickle.dump(model, model_file)
    print(f"Model saved at: {model_path}")

# Save the vectorizer
with open(vectorizer_path, 'wb') as vec_file:
    pickle.dump(cv, vec_file)
    print(f"Vectorizer saved at: {vectorizer_path}")
