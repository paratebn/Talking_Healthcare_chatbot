import speech_recognition as sr
import pyttsx3
import time
import random
import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

# Load data
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Initialize speech
recognizer = sr.Recognizer()
mic = sr.Microphone()
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 1.0)

# Set male voice only
voices = engine.getProperty('voices')
male_voice = None
for voice in voices:
    if 'male' in voice.name.lower():
        male_voice = voice.id
        break
if male_voice:
    engine.setProperty('voice', male_voice)
else:
    engine.setProperty('voice', voices[0].id)

# Speak utility
def speak(text):
    engine.say(text)
    engine.runAndWait()

# NLP functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm sorry, I couldn't understand your symptom."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I don't have a response for that yet."

def calling_the_bot(text):
    prediction = predict_class(text)
    response = get_response(prediction, intents)
    speak(f"From our database, we found that {response}")
    print("You said: ", text)
    print("Bot response: ", response)

# Start bot
print("Bot is Running")
speak("Hello user, I am Bagley, your personal talking healthcare assistant. I am continuing with a male voice.")
print("I am continuing with Male Voice.")

while True:
    try:
        # Ask for symptoms
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            speak("Please tell me your symptoms. I am listening.")
            print("Say Your Symptoms. The Bot is Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=7)
            text = recognizer.recognize_google(audio)
            speak(f"You said: {text}")
            speak("Scanning the database. Please wait.")
            time.sleep(1)
            calling_the_bot(text)
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand your symptom clearly. Please try again.")
        continue
    except sr.WaitTimeoutError:
        speak("I didn’t hear anything. Please speak louder or closer to the mic.")
        continue
    except sr.RequestError:
        speak("Sorry, the recognition service is unavailable right now.")
        break

    # Ask if user wants to continue
    speak("Would you like to continue? Say yes to continue or no to exit.")
    print("Listening for continuation...")

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio = recognizer.listen(source, timeout=4, phrase_time_limit=3)
            user_reply = recognizer.recognize_google(audio).lower()
            print(f"You said: {user_reply}")
    except sr.UnknownValueError:
        speak("Sorry, I couldn't understand that.")
        continue
    except sr.WaitTimeoutError:
        speak("I didn't catch that. Let's try again.")
        continue
    except sr.RequestError:
        speak("Recognition service unavailable. Shutting down.")
        break

    # Check user's choice
    if any(word in user_reply for word in ['no', 'exit', 'false', 'stop', 'quit']):
        speak("Thank you. Take care. Shutting down.")
        print("Bot stopped by user.")
        break
    else:
        speak("Okay, let's continue.")
