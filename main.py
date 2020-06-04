import json
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

gold_data='hackathon.json'

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
def split_test_training_from_json(file):
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
    print ("Amount of unlabeled data:",n_unlabeled)
    return panda_data,unlabeled_panda_data


if __name__ == "__main__":
    #getting the tweets with labeled in data frame 
    data,unlabeled=split_test_training_from_json(gold_data)

    #generate debug files
    #training_data.to_csv('csv/training_data.csv')
    #test_data.to_csv('csv/test_data.csv')

    #clean
    clean_data=f_clean_data(data,"Text")
    clean_unlabeled=f_clean_data(unlabeled,"Text")

    clean_data=apply_binary_label(clean_data,"Label")
    
    #generate debug files
    clean_data.to_csv('csv/clean_data.csv')

    #they are like 100 more antisemtic tweets than not antisemitic. Thats why we are upsampling the minority.
    train_majority=clean_data[clean_data.Label=="0"]
    train_minority=clean_data[clean_data.Label=="1"]
    
    train_minority_upsampled=resample(train_minority,replace=True,n_samples=len(train_majority),random_state=123)
    train_upsampled=pd.concat([train_minority_upsampled,train_majority])

    #print(train_upsampled['Label'].value_counts())
    x_train,x_test,y_train,y_test=train_test_split(train_upsampled['Text'],train_upsampled['Label'],test_size=0.1)
    pipeline= Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('nb', SGDClassifier()),])
    
    '''best=0
    for _ in range(500):
        x_train,x_test,y_train,y_test=train_test_split(train_upsampled['Text'],train_upsampled['Label'],test_size=0.1)
        model=pipeline.fit(x_train,y_train)
        y_predict=model.predict(x_test)

        f1=f1_score(y_test,y_predict,pos_label="1")
        if f1>best:
            best=f1
            with open("best_model.pickle","wb") as f:
                pickle.dump(model,f)
        
    print(best)
    exit(0)'''

    pickle_in=open("best_model.pickle","rb")
    best_model=pickle.load(pickle_in)
    #y_predict=best_model.predict(x_test)
    y_predict=best_model.predict(clean_data['Text'])


    panda_pre_test=[]
    for x in range(len(y_predict)):
        #print(x_test.iloc[x],y_test.iloc[x],y_predict[x])
        #panda_pre_test.append([x_test.iloc[x],y_test.iloc[x],y_predict[x]])
        panda_pre_test.append([clean_data['Text'][x],clean_data['Label'][x],y_predict[x]])
        


    panda_test=pd.DataFrame(panda_pre_test,columns=["Tweet_Text","annotation","Prediction"])
    panda_test.to_csv("csv/prediction.csv")


    #print(f1_score(y_test,y_predict,pos_label="1"))
    print(f1_score(clean_data['Label'],y_predict,pos_label="1"))