import requests
import json



text=' weppe moviebob same white petit bourgeois start openly saying stuff like beating to death niggers and kikes make my dick'

url='https://jnlp.semantic-tech.com/?text='+text
auth=('antisem2020',"6O1O;<JjA=,J#%+w(O&hc>I6_*&zoCe")

r=requests.get(url=url,auth=auth)
r_json=json.loads(r.text)

new_text=""
for i in r_json['documents'][0]['tokenList']:
    if not (i['features']['Stop']):
        new_text=new_text+" "+i['lemma']

print(new_text)
