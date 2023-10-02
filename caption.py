import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import Model,load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers import add

model = load_model("static/model_9.h5")
model.make_predict_function()
model_temp = ResNet50(weights='imagenet',input_shape=(224,224,3))
model_resnet = Model(model_temp.input,model_temp.layers[-2].output)
model_resnet.make_predict_function()


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    feature_vectore = model_resnet.predict(img)
    feature_vectore = feature_vectore.reshape((1,-1))
    return feature_vectore

with open('word2idx.pkl','rb') as f:
        word_to_idx = pickle.load(f)
        
with open('idx2word.pkl','rb') as f:
        idx_to_word = pickle.load(f)

def predict_caption(photo):
    
    max_len = 35
    in_text = 'startseq'
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax()   #greedy sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word=='endseq':
            break
            
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_this_image(image):
    enc = encode_image(image)
    caption = predict_caption(enc)
    return caption

#print(caption_this_image('image.jpg'))
