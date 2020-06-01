import json
import pandas as pd
import numpy as np

gold_data='hackathon.json'

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
    training_data,test_data=split_test_training_from_json(gold_data)
    print(training_data)
    print(test_data)    

