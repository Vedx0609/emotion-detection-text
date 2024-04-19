from flask import Flask,request,render_template,jsonify
import pickle
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords , wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from spellchecker import SpellChecker
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from joblib import load
import tensorflow as tf
import joblib
from csv import writer

# Instantiate LabelEncoder
label_encoder = LabelEncoder()
label_encoder = joblib.load('label_encoder.pkl')

tokenizer = Tokenizer()
tokenizer = joblib.load('tokenizer.pkl',)


nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

def convert_to_lower_case(text):
    return text.lower()

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_stop_words(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])

def get_wordnet_pos(word):
    # Map POS tag to first character that the lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(text)])


def cleaned_sentence(text):
    # text = spell_check(text)
    text = convert_to_lower_case(text)
    text = remove_special_characters(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text


def predict_sentiment(sentence):
    cleaned_sentences = cleaned_sentence(sentence)
    tokenized_sentence = tokenizer.texts_to_sequences([cleaned_sentences])
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=178)
    predicted_emotion = label_encoder.inverse_transform([np.argmax(model.predict(padded_sentence))])[0]
    return predicted_emotion

#df = pd.DataFrame(columns=['Statement','Emotion'])
#df.to_csv('output.csv',columns=['Statement','Emotion'],index=False)



app = Flask(__name__,template_folder='templates')

model = tf.keras.models.load_model("model_tf.keras")  # Load the SavedModel



@app.route('/')
def hello_world():
    return render_template('integrate.html')

@app.route('/sentimental_analaysis', methods = ['POST'])
def sentiment_predict():
    try:
        list1 = []
        Statements = request.form['Statement']

        input_data = {
            'Statement':Statements,
        }

        predicted_emotion = predict_sentiment(Statements)
        print(Statements,":",predicted_emotion)
        output = predicted_emotion

        list1.extend([Statements,output])
        print(list1)
        
        with open('output.csv','a') as file_object:
            writer_object = writer(file_object)
            writer_object.writerow(list1)
            file_object.close()

        return render_template('integrate.html',pred=output)
    except Exception as e:
        return jsonify({'error':str(e)})

if __name__ == "__main__":
    app.run(debug=True)