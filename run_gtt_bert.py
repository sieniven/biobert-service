import os
import json

from Biobert import Biobert
from BiobertModel import BiobertModel

def main():
    models = []
    with open('../gtt_docker/ingest/ingestDataset.json') as f:
        input_data_list = json.loads("[" + f.read().replace("}{", "},{") +  "]")
    
    for data in input_data_list:
        model = BiobertModel(data)
        models.append(model)

    # run Biobert model
    biobert = Biobert(models)
    print("Output predictions from titles: ")
    print(biobert.output_title)
    print("Output predictions from abstract: ")
    print(biobert.output_abstract)

if (__name__ == "__main__"):
    main()
