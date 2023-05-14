# -*- coding: utf-8 -*import streamlit as st-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pickle
import re
#import pysrt
import pandas as pd
import numpy as np
from io import StringIO
from PIL import Image
from catboost import CatBoostClassifier
#from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')

HTML = r'<.*?>' # html тэги меняем на пробел
TAG = r'{.*?}' # тэги меняем на пробел
COMMENTS = r'[\(\[][A-Za-z ]+[\)\]]' # комменты в скобках меняем на пробел
UPPER = r'[[A-Za-z ]+[\:\]]' # указания на того кто говорит (BOBBY:)
LETTERS = r'[^a-zA-Z\'.,!? ]' # все что не буквы меняем на пробел
SPACES = r'([ ])\1+' # повторяющиеся пробелы меняем на один пробел
DOTS = r'[\.]+' # многоточие меняем на точку
SYMB = r"[^\w\d'\s]" # знаки препинания кроме апострофа

def load_tf_idf():
    with open('D:/DS+/9.GIT/dev/English_level/vectorizer.pcl', 'rb') as fid:
        return pickle.load(fid)


def preprocessing(subs):
    subs = subs[1:] # удаляем первый рекламный субтитр
    txt = re.sub(HTML, ' ', subs) # html тэги меняем на пробел
    txt = re.sub(TAG, ' ', txt) # html тэги меняем на пробел
    txt = re.sub(COMMENTS, ' ', txt) # комменты в скобках меняем на пробел
    txt = re.sub(UPPER, ' ', txt) # указания на того кто говорит (BOBBY:)
    txt = re.sub(LETTERS, ' ', txt) # все что не буквы меняем на пробел
    txt = re.sub(DOTS, r'.', txt) # многоточие меняем на точку
    txt = re.sub(SPACES, r'\1', txt) # повторяющиеся пробелы меняем на один пробел
    txt = re.sub(SYMB, '', txt) # знаки препинания кроме апострофа на пустую строку
    txt = re.sub('www', '', txt) # кое-где остаётся www, то же меняем на пустую строку
    txt = txt.lstrip() # обрезка пробелов слева
    txt = txt.encode('ascii', 'ignore').decode() # удаляем все что не ascii символы  
    txt = txt.lower() # текст в нижний регистр
    txt = txt.replace('"', '')
    
#def words_unique(subs):
    txt = txt.split()
    txt_unique = ' '.join(list(set(txt)))

#def lem(row):
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(txt_unique)
    lemmatized_txt = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    
    return lemmatized_txt 


st.title('Определение уровня фильма для изучения английского языка')

#opening the image
image = Image.open('D:\DS+\9.GIT\dev\English_level\image_title.jpg')
#displaying the image on streamlit app
st.image(image, width=500)

uploaded_file = st.file_uploader(label='Загрузите файл с субтитрами в форматe *.srt')
if uploaded_file is not None:
    
    bytes_data = uploaded_file.getvalue()
      
    stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
    
    string_data = stringio.read()
    
    sub_lemm = preprocessing(string_data)
    sub_lemm_text = [sub_lemm] 
        
    tf_idf = load_tf_idf()
    X_values_vect = tf_idf.transform(sub_lemm_text)
    
    model = CatBoostClassifier()     
    model.load_model('D:\DS+\9.GIT\dev\English_level\catboost_model.bin')
    
    y_pred = model.predict(X_values_vect)
    st.write('Уровень сложности фильма: ', y_pred[0,0])


