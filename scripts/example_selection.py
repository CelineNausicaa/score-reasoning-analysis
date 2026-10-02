import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import load_json

path = 'data/test_datasets/deepseek-r1:32b_12samples_verbose_annotationround2_swappingFalse.json'
data = load_json(path, shuffle = True)


sentiments = {"Positive", "Negative", "Neutral"}

examples = {}

for sent in sentiments:
    for feedback in data:
        unique_ID = feedback["unique_ID"]
        if unique_ID not in examples.values():
            if sent not in examples and sent in feedback.values():
                examples[sent] = unique_ID 
print(examples)

examples = {}
for sent in sentiments:
    for feedback in data:
        unique_ID = feedback["unique_ID"]
        #val = examples.values()

        if any(unique_ID in d.values() for d in examples.values()) == False:
            for k, v in feedback.items():
                if v == sent and sent not in examples:
                    corresponding_key = k.replace("sentiment", "reason")
                    corresponding_reason = feedback[corresponding_key]
                    examples[sent] = {"unique_ID": unique_ID, "Example": corresponding_reason}
                
print(examples)
