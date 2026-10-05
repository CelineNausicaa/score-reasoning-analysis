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


def process_test_dataset(test_dataset):
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

    text = "The feedback was clear and very helpful. It has some unclarities, but overall it is good."

    one_shot_examples = [f"\nText: {neutral} Sentiment: Neutral",
                f"\nText: {negative} Sentiment: Negative",
                f"\nText: {positive} Sentiment: Positive"]


    random.shuffle(one_shot_examples)

    prompt = f"""Classify the sentiment of the text as:
                Positive, Negative or Neutral.

                Return only the label. Do not explain your reasoning.

                Here are annotated examples to guide you:

                {"".join(one_shot_examples)}

                Text: {text}
                """
    return prompt

def query_model(prompt):
    class Sentiment(BaseModel):
        sentiment: str

    response = ollama.chat(
        model="deepseek-r1:32b",
        messages=[
            {"role": "user",
            "content": prompt}
        ],
        format = Sentiment.model_json_schema(),
    )

    sentiment = response["message"]["content"].strip()

    return sentiment


def apply_query(processed_test_dataset, examples):
    print("Starting querying process.")

    sentiments = []

    for text in tqdm.tqdm(processed_test_dataset[:10]):

        prompt = build_prompt(text, examples)
        sentiment = query_model(prompt)
        sentiments.append(sentiment)

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
        help="The json file containing the test data"
    )     

    args = parser.parse_args()

    examples_dataset = args.examples_dataset
    test_dataset = args.test_dataset  

    file_name = test_data.split("/")[-1]
    
    outdir = "results/classification/" + file_name

    ### Load and process test data ###
    loaded_test_dataset = load_json(test_dataset)
    processed_test_dataset = process_test_dataset(loaded_test_dataset)

    ### Load and process examples ###
    examples = load_json(examples_dataset)
    
    ### Query LLM ###
    sentiments = apply_query(processed_test_dataset, examples)
    print(sentiments)

    output_json(sentiments, outdir)



if __name__ == "__main__":
    main()
