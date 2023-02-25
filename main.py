from flask import Flask, render_template, request
from googletrans import LANGUAGES, Translator
from textblob import TextBlob
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from mtranslate import translate
import pickle
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from xgboost import XGBClassifier
import speech_recognition as sr
import io
import openai
import os


app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = "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"

openai.api_key = os.environ["OPENAI_API_KEY"]


# Initialize sentiment analyzer
# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()

# Load XGBoost model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
tokenizer = AutoTokenizer.from_pretrained('tokenizer_info')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Home page
@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES)

# Analyze text
@app.route('/analyze', methods=['POST'])
def analyze():
    
    if True:
    #if 'audio' in request.files.keys():
        # Convert audio file to text
        r = sr.Recognizer()
        file = request.files['audio']
        audio = sr.AudioFile(io.BytesIO(file.read()))
        
        with audio as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language = "es-ES")
        
    else:     
        text = request.form['text']
        
    language = request.form['language']
    
    # Translate text to English
    text = translate(text, to_language='en')
    
    # Analyze sentiment
    # scores = sid.polarity_scores(text)
    # sentiment = max(scores, key=scores.get)
    
    def preprocess_input_text(text):
        # Tokenize input text
        encoded_text = tokenizer.encode(text, padding=True, truncation=True, return_tensors='tf')
        encoded_text = encoded_text.numpy()

        # Convert encoded text back into words
        words = [tokenizer.decode([token]) for token in encoded_text[0]]
        input_text = ' '.join(words)
        
        vector = vectorizer.transform([input_text])

        return vector
    
    # Vectorize input text
    vector = preprocess_input_text(text)
    
    # Predict sentiment
    sentiment = model.predict(vector)[0]
    
    emotion_map = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }

    sentiment = emotion_map[sentiment]
    
    # Check for threats
    if sentiment == 'neg' and 'call EMS' in text:
        recommendation = 'Please call EMS immediately!'
    else:
        recommendation = None
    
    def generate_response(prompt):
        response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=60,
        n=1,
        stop=None,
        temperature=0.5,
        )

        message = response.choices[0].text.strip()
        return message
    
    suggested_response = generate_response(text)
    return render_template('result.html', text=text, sentiment=sentiment, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)