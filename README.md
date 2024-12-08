# P3-spam-mail-detector-by-NLP-and-ML
<h2>Table of Contents</h2>
Project Overview
Usage
How It Works

<h2>Project Overview</h2>
<p>The Email Spam Classification application leverages machine learning to classify whether a given email is spam or not. The model is trained on a dataset and uses the Naive Bayes algorithm for classification. The app is built using Streamlit, which provides an easy-to-use interface for the user to input email content and get predictions.</p>
<h2>Usage</h2>
<p>Once the app is running, you will be able to enter an email's text into the provided text box.
Click on the Classify button to classify the email as Spam or Not Spam (Ham).
The classification result will be displayed below the text box.</p>
<h2>How It Works</h2>
<P>Data Preprocessing: The input email text is preprocessed by:

Converting the text to lowercase.
Removing special characters, numbers, and punctuation.
Removing stopwords (common words such as "the", "is", "and", etc.).
Model: The Naive Bayes classifier (MultinomialNB) is trained on a dataset of labeled emails (spam/ham). The classifier is saved as a .pkl file and loaded into the Streamlit app for making predictions.

Vectorization: The CountVectorizer is used to convert the text data into a format that can be fed into the machine learning model.

Prediction: When an email is entered into the text box, the text is preprocessed, vectorized, and passed through the trained model to predict if the email is spam or not.</P>
