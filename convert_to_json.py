import pandas as pd
import json

'''#csv to json
csv_data=pd.read_csv("sheet.tsv",sep="\t",header=0)
k=0
json_data=[]
for i in csv_data['Label']:
    if int(csv_data['Label'][k]==0):
        rating="1"
    else:
        rating="5"

    json_data.append({
        "text":csv_data['Text'][k],
        "antisemitism_rating":rating

    })
    k=k+1

with open("sheet.json","w") as outfile:
    json.dump(json_data,outfile)
'''
#unlabeled json to json
with open ("hackathon.json",'r') as read_file:
        json_data=json.load(read_file)

k=0
pre_unlabeled_panda_data=[]
for i in json_data:
    if not 'antisemitism_rating' in i:
        pre_unlabeled_panda_data.append({
                "text":i['text'],
                "id_str":i['id_str']
            })
        k=k+1
print(k)
with open("unlabled.json","w") as outfile:
    json.dump(pre_unlabeled_panda_data,outfile)      





