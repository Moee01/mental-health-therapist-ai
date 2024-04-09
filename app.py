import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
from googletrans import Translator
import json
import random
from flask import Flask, render_template, request

# Download NLTK data
nltk.download('popular')

# Load model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Lemmatizer
lemmatizer = WordNetLemmatizer()

def tokenize_and_lemmatize(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow_vector(sentence, words, show_details=True):
    sentence_words = tokenize_and_lemmatize(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    if show_details:
        print("found in bag: %s" % ', '.join([w for w, b in zip(words, bag) if b == 1]))
    return np.array(bag)

def predict_class(sentence, model):
    p = create_bow_vector(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg, target_language='en'):
    # Detect the language of the input message
    translator = Translator()
    detected_language = translator.detect(msg).lang
        
        
    
    # Translate the message to English for processing
    if detected_language != 'en':
        msg = translator.translate(msg, dest='en').text
    
    # Perform intent prediction
    ints = predict_class(msg, model)
    
    # Get the response based on the detected intent
    res = get_response(ints, intents)
    
    # Translate the response back to the detected language
    if detected_language != 'en':
        res = translator.translate(res, dest=detected_language).text
       
        print(f"Response: {res}")
    return res


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    return chatbot_response(user_text)

if __name__ == "__main__":
    app.run()
