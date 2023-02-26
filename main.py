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
import re
from gtts import gTTS


app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = ""

openai.api_key = os.environ["OPENAI_API_KEY"]


# sentiment analyzer
# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()

# Load XGBoost model, tokenizer, vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
tokenizer = AutoTokenizer.from_pretrained('tokenizer_info')

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


# Home
@app.route('/')
def home():
    return render_template('index.html', languages=LANGUAGES)

# Analyze text
@app.route('/analyze', methods=['POST'])
def analyze():
    
    if request.form['input'] == 'audio':
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
    
    # Check for threats (fix this, just to check if certain words denoting violence are present)
    threat_regex = r'\b(kill|die|murder|assassinate|destroy)\b'

    if re.search(threat_regex, text):
        recommendation = 'You might be in danger. Please call EMS immediately!'
    else:
        recommendation = None
    
    def generate_response(prompt):
        response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=30,
        n=1,
        stop=None,
        temperature=0.5,
        )

        message = response.choices[0].text.strip()
        return message
    
    string = f"Give me an example response from Person B. Person A: '{text}' Person B: "
    
    # suggested_response = generate_response(string)
    suggested_response = 'what is my name? my name is catherine.'
    suggested_response = suggested_response.split("Person A:")[0]
    suggested_response = suggested_response.strip().replace("'", "")


    suggested_response2 = translate(suggested_response, to_language=language)
    
    language_map = {
        'en': 'english',
        'es': 'spanish',
        'zh': 'chinese (simplified)',
        'fr': 'french',
        'ko': 'korean',
        'de': 'german'
    }

    
    tts = gTTS(text=suggested_response2, lang=language)
    tts.save('audio.mp3')

    
    language = language_map[language]
    
    return render_template('result.html', text=text, language = language, sentiment=sentiment, recommendation=recommendation, message=suggested_response, tr_message = suggested_response2)

if __name__ == '__main__':
    app.run(debug=True)