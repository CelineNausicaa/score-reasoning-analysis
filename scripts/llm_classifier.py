import sys
import os
import argparse
import ollama
import random

from pydantic import BaseModel
from itertools import permutations

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import load_json

example_dataset = "data/selected_examples/gemma3:4b_examples.json"

examples = load_json(example_dataset)

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

print(sentiment)