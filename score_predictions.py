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

with open("./outputs/saved_outputs/predict_abstracts.txt") as f:
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
for idx in len(predicted_titles):
    tmp = BiobertModel()
    tmp.title_entities = predicted_titles[idx]
    tmp.abstract_entities = predicted_abstracts[idx]

for model in models:
    model.get_proteins()
    # model.get_families()