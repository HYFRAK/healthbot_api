from flask import Flask
from flask import  redirect, url_for, request,render_template,redirect
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import Xception
import boto3
import csv

import datetime

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import nltk
from difflib import get_close_matches
import keras
import json



nltk.download('wordnet')
ps = PorterStemmer()








with open('models\sym_dis_map_base.json', 'r') as fire:
  
  dict1 = json.load(fire)

jrr = pd.read_csv('models\Symptom_severity.csv')#change

symp_list_1 = ['itching']
for i in jrr['itching']:
  symp_list_1.append(i)




precautionDictionary = {}
description_list = {}
def getprecautionDict():
    global precautionDictionary
    with open('models\symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def getDescription():
    global description_list
    with open('models\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

getprecautionDict()
getDescription()


def finder(i):
  j = wordnet.synsets(i.lower())
  for k in j:
    k = k.lemma_names()
    for m in k:
      if m.lower() in dict1.keys():
        return dict1[m.lower()]
      elif ps.stem(m) in dict1.keys():
        return dict1[ps.stem(m)]
  j = wordnet.synsets(ps.stem(i))
  for k in j:
    k = k.lemma_names()
    for m in k:
      if m.lower() in dict1.keys():
        return dict1[m.lower()]
  if i.lower() in dict1.keys():
    return dict1[i.lower()]
  elif ps.stem(i) in dict1.keys():
    return dict1[ps.stem(i)]
  z = get_close_matches(i.lower(), list(dict1.keys()),  n=3, cutoff= 0.4)
  if len(z) > 0:
    i = dict1[z[0]]
  return i

def get_response(inpu,user):
  response = client.post_text(botName="HealthCare",botAlias='naman',userId=user,inputText=inpu)
  
  return response['message']




app = Flask(__name__,)




ACCESS_ID="AKIAXUPXIBSDND3UQ5F2"
ACCESS_KEY="RfNF3lP7L2kMBqIiAZvPpw3tzsONNwXZw9JWE9nu"
client = boto3.client('lex-runtime',region_name='us-east-1',aws_access_key_id=ACCESS_ID,aws_secret_access_key= ACCESS_KEY)

sym=False
symptons=[]
err_list = []
user_name=""

model = load_model("models\disease_model.h5")


@app.route("/chat",methods = ['POST'])
def chat():
  global user_name                                                                          
  global sym
  global symptons
  global precautionDictionary
  global description_list

  recieve = request.form.to_dict(flat=False)
  input=recieve['msg'][0]
  user_name=recieve['name'][0]
  print(input,user_name)

  
  response=get_response(input,user_name)
  if response=="Sure, on what date would you like me to schedule your online appointment.":
    return response

  if response=="What are your symptoms?":
    sym=True

  if sym:
    
    if input.lower()!="yes" and input.lower()!="no":
      symptons.append(input)
    if input=="no":
      err_empty = True
      symptons.pop(0)
      k2 = pd.read_csv("models/symp.csv", index_col=0)
      k2 = k2.replace(1,0)
      y2 = pd.read_csv("models/dis.csv", index_col=0)
      symptons_set=set(symptons)
      sym=False
      for inpu in symptons_set:
        inpu2 = inpu.lower().split(' ')
        inpu = finder(inpu2[0])
        
        for i in range(1,len(inpu2)):
          inpu += "_" + finder(inpu2[i])
        if(inpu in symp_list_1):
          k2[inpu] = 1
          
          continue
        z = get_close_matches(inpu, symp_list_1,  n=3, cutoff= 0.8)
        if len(z) > 0:
          
          k2[z[0]] = 1
        else:
          err_list.append(inpu)
          err_empty = False
          #print('symptom not registered, please reframe')
      ans = model.predict(k2)
      j = y2.columns
      ret_json={}

      response = "The Chatbot  predicts your diagnosis to be : " + str(j[np.argmax(ans)]) + " \n \n"

      disease =  str(j[np.argmax(ans)]) 
      ret_json['disease']=disease
      response += description_list[str(j[np.argmax(ans)])] + "\n"
      precution_list=precautionDictionary[str(j[np.argmax(ans)])]
      response += "Take following measures : \n"
      
      prec_list=[]
      for  i,j in enumerate(precution_list):
        
        response += str(i+1) + ")" + str(j) + "\n"
        prec_list.append(j)
      k2 = k2.replace(1,0)

      
      ret_json['prec_list']=prec_list

      return ret_json


      if not err_empty:
        pass
        #response += ' .Symptoms that were not registered are : '.join(err_list)
      symptons.clear()
      err_list.clear()

  return response


app.run()