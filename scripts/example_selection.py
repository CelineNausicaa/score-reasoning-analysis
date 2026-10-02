'''
This script selects one example for each sentiment, given a test dataset.

USAGE
-----
python3 scripts/example_selection.py --test_dataset data/test_datasets/deepseek-r1:32b_12samples_verbose_annotationround2_swappingFal
se.json

OUTPUT
------
a selected set of example, one per sentiment, found under the selected_examples directory
--> outdir: selected_examples/model_name_examples.json
'''

import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import load_json, output_json

def select_examples(data):
    """
    Select an example for each sentiment.
    First, checks if that specific example ID has already been selected.
    We do not want to have the same example feedback repeat itself, so as to take a variety of 1-shot examples

    Parameters
    ----------
    data: a json file object
    """
    sentiments = {"Positive", "Negative", "Neutral"}
    examples = {}
    for sent in sentiments:
        for feedback in data:
            unique_ID = feedback["unique_ID"]
            if any(unique_ID in d.values() for d in examples.values()) == False:
                for k, v in feedback.items():
                    if v == sent and sent not in examples:
                        corresponding_key = k.replace("sentiment", "reason")
                        corresponding_reason = feedback[corresponding_key]
                        examples[sent] = {"unique_ID": unique_ID, "Example": corresponding_reason}
                    
    return examples

def main():

    parser = argparse.ArgumentParser(
        description="Select examples for 1-shot learning"
    )

    parser.add_argument(
        "--test_dataset",
        required=True,
        help="The json file containing the test dataset from which to extract examples"
    )     

    args = parser.parse_args()

    test_dataset = args.test_dataset  

    model_name = test_dataset.replace('data/test_datasets/', "").split("_", 1)[0]

    print("Test dataset:", test_dataset)
    print("Model name:", model_name)

    path = test_dataset
    data = load_json(path, shuffle = True)

    selected_examples = dict(sorted(select_examples(data).items()))
    outdir = "data/selected_examples/" + model_name + "_examples.json"

    output_json(selected_examples, outdir)




if __name__ == "__main__":
    main()