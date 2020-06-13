import pandas as pd
import json


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



