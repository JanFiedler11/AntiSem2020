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

    #getting the tweets with labeled in data frame 
    data,_=pd_from_json('input_data/all.json')

    
    print("Cleaning data from @ , Emojis and Stuff")
    clean_data=f_clean_data(data,"Text")
    
    print("Removing Stop Words and applying lemmatization")
    clean_data=remove_stop_apply_lemma(clean_data,"Text")
    
    clean_data=apply_binary_label(clean_data,"Label")
    
    train_majority=clean_data[clean_data.Label=="0"]
    train_minority=clean_data[clean_data.Label=="1"]
    
    #print(train_majority['Label'].value_counts())
    #print(train_minority['Label'].value_counts())
    
    #they are like 1200 more antisemtic tweets than not antisemitic. Thats why we are upsampling the minority.
    
    train_minority_upsampled=resample(train_minority,replace=True,n_samples=len(train_majority),random_state=123)
    train_upsampled=pd.concat([train_minority_upsampled,train_majority])

    #how much samples we have in total after upsampling
    #print(train_upsampled['Label'].value_counts())

    train_upsampled=shuffle(train_upsampled)

    x_train,x_test,y_train,y_test=train_test_split(train_upsampled['Text'],train_upsampled['Label'],test_size=0.2)
    pipeline=Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('nb', SGDClassifier()),])
    
    #We can either let the model train live once
    model=pipeline.fit(x_train,y_train)

    #Or save the model with the best f1 score (n iterations) and load it later 
    #save_best_model(train_upsampled,pipeline,1000,"best_model.pickle")
    
    #load the pickle if you want to use the best model
    #pickle_in=open("best_model.pickle","rb")
    #model=pickle.load(pickle_in)
    

    #predict the results
    y_predict=model.predict(x_test)

    print("Data from test gold standard","-"*40)
    print("F1 - Score: ",f1_score(y_test,y_predict,pos_label="1"))
    print("Accuracy: ",accuracy_score(y_test,y_predict))
    print("-"*60)


    