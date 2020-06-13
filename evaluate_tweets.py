import json
import pandas as pd
import numpy as np
import sys
import re
import pickle
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

'''
convert the predictions and Ids into 1 df in order to save it to csv
'''
def df_from_predictions_and_Ids(Ids,predictions):
    pre_csv_data=[]
    k=0
    for i in predictions:
        pre_csv_data.append([Ids[k],i])
        k=k+1
    
    results=pd.DataFrame(pre_csv_data,columns=["Tweet Id","Evaluation"])
    return results


'''
remove stop words from tweets and apply lemmatization for 1 string
'''
def remove_stop_apply_lemma_for_string(text):
    new_text=""
    tokens=nlp(text)
    for token in tokens:
        if not token.is_stop:
            new_text=new_text+" "+token.lemma_
    
    return new_text

'''
remove stop words from tweets and apply lemmatization for 1 DataFrame
'''
def remove_stop_apply_lemma(df,field):
    k=0
    for i in df[field]:
        if(k % 100==0):
            print(".",end="",flush=True)
        df[field][k]=remove_stop_apply_lemma_for_string(i)
        k=k+1
    print("")
    return df


'''
clean the text data from @users, emojis and other stuff
'''
def f_clean_data(df,field):
    df[field]=df[field].str.lower()
    df[field] = df[field].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df

'''
apply a binary filter for the gold data.
antisemtic level: 1-3 = not antisemitic
antisemtic level: 4-5 = antisemitic
'''
def apply_binary_label(df,field):
    k=0
    for i in df[field]:
        if int(i)>3:
            df[field][k]="1"
        else:
            df[field][k]="0"
        k=k+1
    return df

'''
split the file to read from json into a panda dataframe
'''
def df_from_json(file):
    with open (file,'r') as read_file:
        json_data=json.load(read_file)

    k=0
    l=0
    pre_panda_data=[]
    for i in json_data:
        if 'text' in i and 'id_str' in i:
            pre_panda_data.append([i['text'],i['id_str']])
            k=k+1
        else:
            l=l+1
        
    panda_data=pd.DataFrame(pre_panda_data,columns=["Text","Id"])
    print ("Amount of tweets:",k+l)
    print ("Amout of tweets with either no text or id",l)
    return panda_data


if __name__ == "__main__":
    print("******************************************")
    print("Welcome to the Evaluation Model of Group 2")
    print("******************************************\n")


    input_file=input("Enter the path/name of the json file to read\n")
    
    try:    
        gold_data_ev=df_from_json(input_file)
        print("successful file read")
    except FileNotFoundError:
        print("path/name of the json file does not exist - exiting")
        exit(0)
    
    #Loading Spacy
    nlp=spacy.load('en_core_web_sm')
    
    #load the trained model as pickle
    pickle_in=open("best_model.pickle","rb")
    model=pickle.load(pickle_in)

    print("cleaning the data, removing stop words and applying lemmatization")
    #clean the tweet text from emojis and stuff
    gold_data_ev=f_clean_data(gold_data_ev,"Text")

    #remove stop words and apply lemma
    gold_data_ev=remove_stop_apply_lemma(gold_data_ev,'Text')

    #feed the data into the trained model - get predictions
    gold_data_ev_predict=model.predict(gold_data_ev['Text'])

    #put it into one df
    results=df_from_predictions_and_Ids(gold_data_ev['Id'],gold_data_ev_predict)
    
    #write results into csv
    results.to_csv("results/group2_evaluation.csv")
    print("Done")
