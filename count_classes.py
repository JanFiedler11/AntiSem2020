import json


with open ("group2.json",'r') as read_file:
    data=json.load(read_file)


score1=0
score2=0
score3=0
score4=0
score5=0
some_score=0
no_label=0
k=0
for i in data:
    k=k+1
    try:
        score=i['antisemitism_rating']
        if(score=='1'):
            score1=score1+1
        elif(score=='2'):
            score2=score2+1
        elif(score=='3'):
            score3=score3+1
        elif (score=='4'):
            score4=score4+1
        elif score=='5':
            score5=score5+1
    except:
        no_label=no_label+1


print("Amount of tweets with ranking 1:",score1)
print("Amount of tweets with ranking 2:",score2)
print("Amount of tweets with ranking 3:",score3)
print("Amount of tweets with ranking 4:",score4)
print("Amount of tweets with ranking 5:",score5)

print("-"*40)
print("Amount of unlabeled data:",no_label)

print("Total Data:",k)

