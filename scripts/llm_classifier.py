"""
USAGE
-----
python3 scripts/llm_classifier.py 
--example_dataset path/to/dataset
--test_dataset path/th/dataset
--model model

EXAMPLE USAGE - TESTING
-------------
python3 scripts/llm_classifier.py --examples_dataset data/selected_examples/gemma3:12b_examples.json --test_dataset data/test_datasets/gemma3:12b_12samples_verbose_annotationround2_swappingFalse.json --model deepseek-v2:16b

EXAMPLE USAGE - ANNOTATING
-------------
python3 scripts/llm_classifier.py --examples_dataset data/selected_examples/gemma3:12b_examples.json --test_dataset data/full_datasets/gemma3:12b_Nonesamples_verbose_full1_swappingFalse.json --model magistral:24b --n 500
"""

import sys
import os
import argparse
import ollama
import random
import tqdm


from pydantic import BaseModel
from itertools import permutations


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import *

def process_test_dataset_reason(test_dataset):
    test_data = []
    for dct in test_dataset:
        for k, v in dct.items():
            if "reason_" in k:
                test_data.append(v)
    
    #random.shuffle(test_data)
    
    return test_data

def build_prompt(text, examples):
    negative = examples["Negative"]["Example"]
    neutral = examples["Neutral"]["Example"]
    positive = examples["Positive"]["Example"]

    #text = "The feedback was clear and very helpful. It has some unclarities, but overall it is good."

    one_shot_examples = [f"\n A neutral text contains a mix of positive and negative elements.\nText: {neutral} Sentiment: Neutral",
                f"\nA negative text contains mostly or only negative elements. \nText: {negative} Sentiment: Negative",
                f"\nA positive text contains mostly or only positive elements. \nText: {positive} Sentiment: Positive"]


    random.shuffle(one_shot_examples)

    prompt = f"""
                You are a sentiment analysis classifier.
                Assign each text one of the following sentiment:
                Positive, Negative or Neutral.

                Return only the label. Do not explain your reasoning.

                Here are definitions and annotated examples to guide you:

                {"".join(one_shot_examples)}

                Text: {text}
                """
    return prompt, text

def query_model(prompt, model):
    class Sentiment(BaseModel):
        sentiment: str

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user",
            "content": prompt}
        ],
        format = Sentiment.model_json_schema(),
    )

    sentiment = response["message"]["content"].strip()

    return sentiment


def apply_query(processed_test_dataset, examples, model, outdir):
    print("Starting querying process.")

    sentiments = []

    for text in tqdm.tqdm(processed_test_dataset):

        prompt, reasoning = build_prompt(text, examples)
        sentiment = query_model(prompt, model)
        sentiments.append(sentiment)
        append_jsonl(json.loads(sentiment), outdir)

    return sentiments


    
def main():

    parser = argparse.ArgumentParser(
        description="Select examples for 1-shot learning"
    )

    parser.add_argument(
        "--examples_dataset",
        required=True,
        help="The json file containing the extracted examples"
    )     

    parser.add_argument(
        "--test_dataset",
        required=True,
        help="The json file containing the data to annotate"
    )     

    parser.add_argument(
        "--model",
        required=True,
        help="The selected model"
    )

    parser.add_argument(
        "--n",
        required=False,
        type = int,
        help="If specified, only the first n datapoints are taken."
    )          

    args = parser.parse_args()

    examples_dataset = args.examples_dataset
    test_dataset = args.test_dataset  
    model = args.model
    n = args.n

    print("Examples taken from:", examples_dataset)
    print("Dataset to be annotated is:", test_dataset)
    print("Model selected is:", model)

    file_name = test_dataset.split("/")[-1]
    
    outdir = "results/classification/" + model + "_" + file_name

    ### Load and process test data ###
    loaded_test_dataset = is_json_or_jsonl(test_dataset)
    processed_test_dataset = process_test_dataset_reason(loaded_test_dataset)

    if n:
        print(f"Taking only the first {n} datapoints.")
        processed_test_dataset = processed_test_dataset[:n]

    print("First element to annotate is:", processed_test_dataset[0])
    print("Last element to annotate is:", processed_test_dataset[-1])

    ### Load and process examples ###
    examples = is_json_or_jsonl(examples_dataset)
    
    ### Query LLM ###
    sentiments = apply_query(processed_test_dataset, examples, model, outdir)

    #processed_sentiments = [json.loads(s) for s in sentiments]

    #output_json(processed_sentiments, outdir)

if __name__ == "__main__":
    main()
