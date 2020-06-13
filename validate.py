import json
import pandas as pd
import numpy as np
import re
import pickle
import json
import spacy
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.metrics import f1_score,accuracy_score

'''
save the best model based on the f1 score to a pickle file
'''
def save_best_model(train_upsampled,pipeline,iterations,filepath):
    best=0
    for _ in range(iterations):
        x_train,x_test,y_train,y_test=train_test_split(train_upsampled['Text'],train_upsampled['Label'],test_size=0.1)
        model=pipeline.fit(x_train,y_train)
        y_predict=model.predict(x_test)

        f1=f1_score(y_test,y_predict,pos_label="1")
        if f1>best:
            best=f1
            with open(filepath,"wb") as f:
                pickle.dump(model,f)
        
    print(best)

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
split the gold standard from json into a panda dataframe
'''
def pd_from_json(file):
    with open (file,'r') as read_file:
        json_data=json.load(read_file)

    n_unlabeled=0
    pre_panda_data=[]
    pre_unlabeled_panda_data=[]
    for i in json_data:
        if 'antisemitism_rating' in i:
            pre_panda_data.append([i['antisemitism_rating'],i['text']])
        else:
            pre_unlabeled_panda_data.append([i['text']])
            n_unlabeled=n_unlabeled+1

    panda_data=pd.DataFrame(pre_panda_data,columns=["Label","Text"])
    unlabeled_panda_data=pd.DataFrame(pre_unlabeled_panda_data,columns=["Text"])
    #print ("Amount of unlabeled data:",n_unlabeled)
    return panda_data,unlabeled_panda_data


if __name__ == "__main__":
    #Loading Spacy
    nlp=spacy.load('en_core_web_sm')
    #load the pickle if you want to use the best model
    pickle_in=open("best_model.pickle","rb")
    model=pickle.load(pickle_in)


    '''while True:
        read_in_data=[]
        my_input=input("Enter a Text\n")
        read_in_data.append([my_input])
        input_data=pd.DataFrame(read_in_data,columns=['Text'])
        input_data=f_clean_data(input_data,'Text')
        input_data=remove_stop_apply_lemma(input_data,'Text')
        input_data_x=input_data['Text']

        print(model.predict(input_data_x))'''



    #lets test the model with some unseen tweets with manual label
    sheet_data=pd.read_csv("input_data/sheet.tsv",sep="\t",header=0)

    sheet_data=f_clean_data(sheet_data,'Text')
    sheet_data=remove_stop_apply_lemma(sheet_data,'Text')
    sheet_y=sheet_data['Label'].astype(str)
    sheet_x=sheet_data['Text']

    y_predict=model.predict(sheet_x)

    print("Data from csv sheet","-"*40)
    print("F1 - Score: ",f1_score(sheet_y,y_predict,pos_label="1"))
    print("Accuracy: ",accuracy_score(sheet_y,y_predict))
    print("-"*60)

    #lets test the model with unseen tweets from the datathon - annotation might not be 100 % right

    group_data,_=pd_from_json('input_data/group2.json')
    group_data=apply_binary_label(group_data,'Label')

    group_data=f_clean_data(group_data,'Text')
    group_data=remove_stop_apply_lemma(group_data,'Text')

    group_y=group_data['Label']
    group_x=group_data['Text']



    y_predict=model.predict(group_x)

    print("Data from group2 json","-"*40)
    print("F1 - Score: ",f1_score(group_y,y_predict,pos_label="1"))
    print("Accuracy: ",accuracy_score(group_y,y_predict))
    print("-"*60)


    gold_data,_=pd_from_json('input_data/hackathon.json')
    gold_data=apply_binary_label(gold_data,'Label')

    gold_data=f_clean_data(gold_data,'Text')
    gold_data=remove_stop_apply_lemma(gold_data,'Text')

    gold_y=gold_data['Label']
    gold_x=gold_data['Text']



    y_predict=model.predict(gold_x)

    print("Data from gold json","-"*40)
    print("F1 - Score: ",f1_score(gold_y,y_predict,pos_label="1"))
    print("Accuracy: ",accuracy_score(gold_y,y_predict))
    print("-"*60)