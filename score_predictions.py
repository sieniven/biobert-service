import os
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
with open(path, 'a', encoding="utf-8") as wf:
    count = 1
    for model in models:
        model.get_proteins()
        model.get_families()
        print("I love denise beh")
        
        entry = str(count) + ":\n" + "abstract protein:\n" + str(model.abstract_proteins) + "\ntitle protein:\n" + \
            str(model.title_proteins) + "\ntitle families:\n" + str(model.title_families) + "\nabstract families:\n" + \
            str(model.abstract_families)
        wf.write(entry+'\n\n')
        count += 1    
    wf.close()