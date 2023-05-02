import json

f = open("captions.json")
data = json.load(f)


for i in data:
    print(i['mid'])

f.close()
