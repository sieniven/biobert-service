import os

tp_biobert = 0
fp_biobert = 0
fn_biobert = 0
tp_biobert_with_grounding = 0
fp_biobert_with_grounding = 0
fn_biobert_with_grounding = 0
tp_reach = 0
fp_reach = 0
fn_reach = 0

path = os.path.join("C:/Users/sieni/biobert-service/outputs/saved_outputs/scores.txt")
with open(path) as f:
    contents = f.readlines()
    for num, line in enumerate(contents):
        index = line.find(":") + 2
        line = line[index:]
        line = line.strip()        
        words = line.split(" ")

        tp_biobert += int(words[0])
        fp_biobert += int(words[1])
        fn_biobert += int(words[2])
        tp_biobert_with_grounding += int(words[3])
        fp_biobert_with_grounding += int(words[4])
        fn_biobert_with_grounding += int(words[5])
        tp_reach += int(words[6])
        fp_reach += int(words[7])
        fn_reach += int(words[8])

precision_biobert = tp_biobert / float(tp_biobert + fp_biobert)
recall_biobert = tp_biobert / float(tp_biobert + fn_biobert)
f1_measure_biobert = 2 * float(precision_biobert * recall_biobert) / float(precision_biobert + recall_biobert)
print(f"Biobert - Precision: {precision_biobert}, Recall: {recall_biobert}, F1-Measure: {f1_measure_biobert}")

precision_biobert_with_grounding = tp_biobert_with_grounding / float(tp_biobert_with_grounding + fp_biobert_with_grounding)
recall_biobert_with_grounding = tp_biobert_with_grounding / float(tp_biobert_with_grounding + fn_biobert_with_grounding)
f1_measure_biobert_with_grounding = 2 * float(precision_biobert_with_grounding * recall_biobert_with_grounding) / float(precision_biobert_with_grounding + recall_biobert_with_grounding)
print(f"Biobert with grounding - Precision: {precision_biobert_with_grounding}, Recall: {recall_biobert_with_grounding}, F1-Measure: {f1_measure_biobert_with_grounding}")

precision_reach = tp_reach / float(tp_reach + fp_reach)
recall_reach = tp_reach / float(tp_reach + fn_reach)
f1_measure_reach = 2 * float(precision_reach * recall_reach) / float(precision_reach + recall_reach)
print(f"Reach - Precision: {precision_reach}, Recall: {recall_reach}, F1-Measure: {f1_measure_reach}")