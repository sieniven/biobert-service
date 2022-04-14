import os
import sys

sys.path.append("../Source/")
from BiobertModel import BiobertModel

predicted_titles = []
predicted_abstracts = []

with open("./outputs/saved_outputs/predict_title.txt") as f:
    contents = f.readlines()
    for line in contents:
        index = line.find(":") + 2
        line = line[index:]
        line = line.strip()
        
        if line:
            words = line.split(" ")
            predicted_titles.append(words)
        else:
            predicted_titles.append([])

with open("./outputs/saved_outputs/predict_abstract.txt") as f:
    contents = f.readlines()
    for line in contents:
        index = line.find(":") + 2
        line = line[index:]
        line = line.strip()
        
        if line:
            words = line.split(" ")
            predicted_abstracts.append(words)
        else:
            predicted_abstracts.append([])

models = []
for idx in range(len(predicted_titles)):
    tmp = BiobertModel()
    tmp.title_entities = predicted_titles[idx]
    tmp.abstract_entities = predicted_abstracts[idx]
    models.append(tmp)



path = os.path.join("C:/Users/sieni/biobert-service/outputs/biobert_predictions.txt")
count = 1
for idx, model in enumerate(models):
    model.get_proteins()
    model.get_families()
    print("I love denise beh")
    
    with open(path, 'a', encoding="utf-8") as wf:
        entry = str(idx + 1) + ":\n"
        entry = entry + "\ntitle protein:\n"
        for k, v in model.title_proteins.items():
            entry = entry + str(k) + " "
            entry = entry + str(v["symbol"]) + " "

        entry = entry + "\nabstract protein:\n"
        for k, v in model.abstract_proteins.items():
            entry = entry + str(k) + " "
            entry = entry + str(v["symbol"]) + " "

        entry = entry + "\ntitle families:\n"
        for k, v in model.title_families.items():
            entry = entry + str(k) + " "
            entry = entry + str(v["symbol"]) + " "
        
        entry = entry + "\nabstract families:\n"
        for k, v in model.abstract_families.items():
            entry = entry + str(k) + " "
            entry = entry + str(v["symbol"]) + " "

        wf.write(entry+'\n\n')
    count += 1    