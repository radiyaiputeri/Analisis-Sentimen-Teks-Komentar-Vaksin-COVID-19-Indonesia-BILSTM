from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global graph,model


model = None

def load_model1(export_path):
    global model


    model1 = load_model('model_pr025_L50_100_nadam.h5')

    #graph = tf.compat.v1.get_default_graph()

#########################Code for Sentiment Analysis
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("home.html")

def hasil(X_opini):
    hasil = model.predict(X_opini)
    pos = hasil[0][0]
    net = hasil[0][1]
    neg = hasil[0][2]

    sentiment=''
    if neg>pos and neg>net:
        sentiment = 'Negative'                  
    elif net>pos and net>neg:
        sentiment = 'Netral'                   
    elif pos>net and pos>neg:
        sentiment = 'Positive'
        
        return sentiment


@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        
        
        
        datavaksin = pd.read_csv("datavaksinlabelprocessed(belum seimbang).csv")
            
       

        text =request.form['text']
        opini = [text]
        sentiment = ''
        #datavaksin = pd.read_csv('daata143new2.csv')
        MAX_VOCAB_SIZE = 2000 # the maximum amount of possible words/tokens
        MAX_SEQUENCE_LENGTH = 50 # in practice, the maximum amount of words per sentence
        tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE, split=' ')
        tokenizer.fit_on_texts(datavaksin['detoken_stem'].values)
        
        X_opini = tokenizer.texts_to_sequences(opini)
        X_opini = pad_sequences(X_opini, maxlen = MAX_SEQUENCE_LENGTH)
        model1 = load_model('model_pr025_L50_100_nadam.h5')
        hasil = model1.predict(X_opini)
        
        
        pos = hasil[0][0]
        net = hasil[0][1]
        neg = hasil[0][2]
        if neg>pos and neg>net:
            sentiment = 'Negative'
        elif net>pos and net>neg:
            sentiment = 'Netral'
        elif pos>net and pos>neg:
            sentiment = 'Positive'
                    
            
    return render_template('home.html',
                            text=text, 
                            sentiment1=sentiment,
                            prob_pos= pos, 
                            prob_net=net, 
                            prob_neg=neg) 
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    load_model1("./")
    app.run(debug=True)
