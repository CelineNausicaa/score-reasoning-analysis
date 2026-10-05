"""
USAGE
-----
python3 scripts/eval.py --test_dataset data/test_datasets/gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json --llm_predictions results/classification/gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json
EXAMPLE USAGE
-------------
python3 scripts/eval.py --test_dataset data/test_datasets/gemma3:4b_12samples_verbose_annotationround2_swappingFalse.json --llm_predictions results/classification/gemma3:4b_12samples_verbose_annotationround2_swappingFalse.json
"""

import sys
import os
import argparse

from sklearn.metrics import classification_report

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import *

def process_test_dataset(loaded_test_dataset):
    y_true = []
    for dct in loaded_test_dataset:
        for k, v in dct.items():
            if "sentiment_" in k:
                y_true.append(v)
    return y_true

def process_llm_predictions(loaded_predictions):
    y_pred = []
    for dct in loaded_predictions:
        y_pred.append(dct["sentiment"])
    return y_pred

def main():

    parser = argparse.ArgumentParser(
        description="Select examples for 1-shot learning"
    )

    parser.add_argument(
        "--test_dataset",
        required=True,
        help="The json file containing the true sentiments"
    )     

    parser.add_argument(
        "--llm_predictions",
        required=True,
        help="The json file containing the llm-annotated sentiments"
    )     

    args = parser.parse_args()

    test_dataset = args.test_dataset
    llm_predictions = args.llm_predictions 

    file_name = test_dataset.split("/")[-1]

    loaded_test_dataset = load_json(test_dataset)
    loaded_predictions = load_jsonl(llm_predictions)

    y_true = process_test_dataset(loaded_test_dataset)
    y_pred = process_llm_predictions(loaded_predictions)

    print(classification_report(y_true, y_pred))

    
    

if __name__ == "__main__":
    main()