import streamlit as st
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib

model = joblib.load('MNB_best_spam.joblib')
vectorizer = joblib.load('vectorizer.joblib')

st.title('Spam or Ham Detection')
st.write('Enter below to predict if its spam or ham')
input_text = st.text_input('Your message')

if st.button("Predict"):
    if input_text:
        # Transform input
        input_vec = vectorizer.transform([input_text])
        # Predict
        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec).max()

        # Display results
        st.write(f"Prediction: **{prediction.upper()}**")
        st.write(f"Confidence: {probability:.2f}")

        # Progress bar (inside same block)
        st.progress(int(probability * 100))
    else:
        st.warning("Please enter a message to predict!")
