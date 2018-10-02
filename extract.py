import json
import pandas as pd

q=[]
a=[]
with open('train.json', "r") as sentences_file:
    reader = json.load(sentences_file)
    for item in reader['data']:
        if type(item)==dict:
            for qas in item['paragraphs']:
                for question in qas['qas']:
                    try:
                        a.append(question['answers'][0]['text'])
                        q.append(question['question'])
                        
                    except:
                        pass
                    break

print(len(q))
print(len(a))
print(q[0:50])
print(a[0:50])    
"""
print(dir(pd))

a=pd.read_json('train.json')
print(a)
"""
"""
with open('train.json', "r") as sentences_file:
    reader = json.load(sentences_file)
    for item in reader['data']:
        if type(item)==dict:
            print(item['paragraphs'][0]['qas'][0]['answers'][0]['text'])
            break

"""
