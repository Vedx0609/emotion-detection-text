# This is the converted .py file from the Jupyter Notebook
# This script is ran once to train the model and save it as a keras file
# The model is then loaded in the UI.py file to predict the emotion of the user input

import pandas as pd
import numpy as np
import joblib
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords , wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#from spellchecker import SpellChecker
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
from joblib import dump

from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

initial_dataset = pd.read_csv('isear.csv')
initial_dataset.head()

# Fetching all the emotions and it's count present in the dataset
emotions = initial_dataset['Emotions'].value_counts()
print(emotions)
print(f"Total entries in the dataset : {emotions.sum()}")

# To check if there are any null values present in the dataset
print(initial_dataset.isnull().sum())

# To check if there are any duplicate values present in the dataset
print(initial_dataset["Statements"].duplicated().sum())

# Displaying all the duplicates present in the dataset
duplicates = initial_dataset[initial_dataset.duplicated()]

#Print all the unique duplicates present in the dataset
print(duplicates.value_counts())

null_responses = ["[ No response.]","[ Do not know.]","NO RESPONSE.","Doesn't apply.","Does not apply.","[ Can not think of anything just now.]" , "[ No description.]" , "[ Never felt the emotion.]" , "[ I have never felt this emotion.]" , "[ Never experienced.]" , "[ Never.]" , "[ Do not remember any incident.]"]

# Removing the rows containing null responses
null_removed_dataset = initial_dataset[~initial_dataset['Statements'].isin(null_responses)]

# To check if there are any null values present in the dataset
print(null_removed_dataset.isnull().sum())

# Fetching all the emotions and it's count present in the dataset
updated_emotions = null_removed_dataset['Emotions'].value_counts()
print(updated_emotions)
print(f"Total entries in the dataset : {updated_emotions.sum()}")

# To check if there are any duplicate values present in the dataset
print(null_removed_dataset["Statements"].duplicated().sum())

# Displaying all the duplicates present in the dataset
updated_duplicates = null_removed_dataset[null_removed_dataset.duplicated()]

#Print all the unique duplicates present in the dataset
print(updated_duplicates.value_counts())

print(updated_duplicates)

# Percentage of emotions present in the dataset
emotions_percentage = updated_emotions/sum(updated_emotions)*100
print(emotions_percentage)

# Plot the emotions vs it's count as a bar plot and keep the scale of count from 1050 to 1100
plt.figure(figsize=(10,5))
plt.ylim(1050, 1100)
sns.countplot(x='Emotions', data=null_removed_dataset)

# Data Cleaning

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

import nltk
nltk.download('averaged_perceptron_tagger')



import nltk
nltk.download('punkt')

#apply preprocessing to the Statement column
null_removed_dataset['Cleaned_Statements'] = null_removed_dataset['Statements'].apply(cleaned_sentence)

#label encoding for emotions
label_encoder = LabelEncoder()
null_removed_dataset['Encoded_Emotion'] = label_encoder.fit_transform(null_removed_dataset['Emotions'])

joblib.dump(label_encoder, 'label_encoder.pkl')

#tokenizing words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(null_removed_dataset['Cleaned_Statements'])
X = tokenizer.texts_to_sequences(null_removed_dataset['Cleaned_Statements'])
X = pad_sequences(X)

joblib.dump(tokenizer, 'tokenizer.pkl')

# train test split stuff
X_train, X_test, y_train, y_test = train_test_split(X, null_removed_dataset['Encoded_Emotion'], test_size=0.2, random_state=42)

#build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
model.add(LSTM(units=128))
model.add(Dense(units=len(null_removed_dataset['Emotions'].unique()), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train the dude
model.fit(X_train, y_train, batch_size=64, epochs=16, validation_data=(X_test, y_test))

#predict sentiment/emotion
def predict_sentiment(sentence):
    cleaned_sentences = cleaned_sentence(sentence)
    tokenized_sentence = tokenizer.texts_to_sequences([cleaned_sentences])
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=X.shape[1])
    predicted_emotion = label_encoder.inverse_transform([np.argmax(model.predict(padded_sentence))])[0]
    return predicted_emotion

input = "I am very angry"

predicted_emotion = predict_sentiment(input)
print(f"Sentence: {input}\nPredicted Emotion: {predicted_emotion}")

input = "I am very happy"

predicted_emotion = predict_sentiment(input)
print(f"Sentence: {input}\nPredicted Emotion: {predicted_emotion}")

model.save("model_tf.keras")  # Save model in HDF5 format
  # Save model in TensorFlow's SavedModel format
