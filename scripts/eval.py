"""
USAGE
-----
python3 scripts/eval.py --test_dataset data/test_datasets/gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json --llm_predictions results/classification/gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json

EXAMPLE USAGE
-------------
python3 scripts/eval.py --test_dataset data/test_datasets/gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json --llm_predictions results/classification/deepseek-v2:16b_gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json
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

def append_file(classifier_name, llm_name, report):
    with open("evaluation_report.txt", "a") as text_file:
        text_file.write("\n" + classifier_name  + " - " + llm_name + "\n" + report)

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

    llm_name = test_dataset.split("_")[1].split("/")[-1]
    llm_predictions_name = llm_predictions.split("_")[1]

    if llm_name != llm_predictions_name:
        print("ERROR. The wrong test dataset has been chosen.")
    
    else:
        classifier_name = llm_predictions.split("_")[0].split("/")[-1]

        file_name = test_dataset.split("/")[-1]

        loaded_test_dataset = load_json(test_dataset)
        loaded_predictions = load_jsonl(llm_predictions)

        y_true = process_test_dataset(loaded_test_dataset)
        y_pred = process_llm_predictions(loaded_predictions)

        report = classification_report(y_true, y_pred)

        append_file(classifier_name, llm_name, report)
    

if __name__ == "__main__":
    main()