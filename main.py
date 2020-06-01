import json
import pandas as pd
import numpy as np
import re
from sklearn.utils import resample

gold_data='hackathon.json'

'''
clean the text data from @users, emojis and other stuff
'''
def clean_data(df,field):
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
split the gold standard from json into test and training in a panda dataframe
'''
def split_test_training_from_json(file):
    with open (file,'r') as read_file:
        data=json.load(read_file)

    # 10 % for testing; 90 % for training
    testing_amount=int(0.1*len(data))

    json_test_data=data[:testing_amount]
    json_training_data=data[testing_amount:]

    n_unlabeled=0
    pre_panda_test_data=[]
    for i in json_test_data:
        if 'antisemitism_rating' in i:
            pre_panda_test_data.append([i['antisemitism_rating'],i['text']])
        else:
            n_unlabeled=n_unlabeled+1

    pre_panda_training_data=[]
    for k in json_training_data:
        if 'antisemitism_rating' in k:
            pre_panda_training_data.append([k['antisemitism_rating'],k['text']])
        else:
            n_unlabeled=n_unlabeled+1


    panda_test_data=pd.DataFrame(pre_panda_test_data,columns=["Label","Text"])
    panda_training_data=pd.DataFrame(pre_panda_training_data,columns=["Label","Text"])
    print ("Amount of unlabeled data:",n_unlabeled)
    return [panda_training_data,panda_test_data]


if __name__ == "__main__":
    #getting the tweets with labeled in data frame for test and training
    training_data,test_data=split_test_training_from_json(gold_data)

    #generate debug files
    #training_data.to_csv('csv/training_data.csv')
    #test_data.to_csv('csv/test_data.csv')

    #clean
    clean_training=clean_data(training_data,"Text")
    clean_test=clean_data(test_data,"Text")

    clean_training=apply_binary_label(clean_training,"Label")
    clean_test=apply_binary_label(clean_test,"Label")

    #generate debug files
    clean_training.to_csv('csv/clean_training.csv')
    clean_test.to_csv('csv/clean_test.csv')

    #they are like 100 more antisemtic tweets than not antisemitic. Thats why we are upsampling the minority.
    train_majority=clean_training[clean_training.Label=="0"]
    train_minority=clean_training[clean_training.Label=="1"]
    
    train_minority_upsampled=resample(train_minority,replace=True,n_samples=len(train_majority),random_state=123)
    train_upsampled=pd.concat([train_minority_upsampled,train_majority])

    print(train_upsampled['Label'].value_counts())
    
    #exit(0)
    print("done")
