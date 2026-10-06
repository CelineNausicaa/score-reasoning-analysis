import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.io import *

def separate_sentiments(full_dataset, n = False):
    """
    Separate the scores and the sentiments into two separated lists.
    The scores are then converted into sentiments to compare with the actual sentiment (reasoning sentiment)
    """

    scores = []
    sentiments = []

    for entry in full_dataset:
        for k, v in entry.items():
            if "score_" in k:
                scores.append(v)
    
    sentiments = ["Positive" if x > 0.6 else "Negative"  if x < 0.5 else "Neutral" for x in scores]

    if n:
        sentiments = sentiments[:n]
        scores = scores[:n]

    return sentiments, scores

def check_sentiment_errors(sentiments):
    """
    Ensures that the data does not contain error.
    I.e. checks that only the labels "neutral", "positive", "negative" are used.
    """

    valid_sentiment = ["Neutral", "Positive", "Negative"]

    for i,sentiment in enumerate(sentiments):
        if sentiment not in valid_sentiment:
            raise ValueError("Invalid sentiment: ", sentiment, i)

def main():

    parser = argparse.ArgumentParser(
        description="Select examples for 1-shot learning"
    )

    parser.add_argument(
        "--llm_predictions",
        required=True,
        help="The json file containing the llm predictions"
    )     

    parser.add_argument(
        "--full_dataset",
        required=True,
        help="The json file containing the data to annotate"
    )    

    parser.add_argument(
        "--n",
        required=False,
        type = int,
        help="If specified, only the firs tn datapoints are taken. You must specify the same n as specified in llm_classifier.py"
    )       

    args = parser.parse_args()

    llm_predictions = args.llm_predictions
    full_dataset = args.full_dataset  
    n = args.n

    print("LLM predictions:", llm_predictions)
    print("Full dataset:", full_dataset)

    ###Ensuring the right datasets are being compared###

    llm_name = full_dataset.split("_")[1].split("/")[-1]
    llm_predictions_name = llm_predictions.split("_")[1]

    if llm_name != llm_predictions_name:
        print("ERROR. The wrong test dataset has been chosen.")

    ### Loading the data ###
    loaded_llm_predictions = is_json_or_jsonl(llm_predictions)
    loaded_full_dataset = is_json_or_jsonl(full_dataset)

    ###Extracting the sentiments from the scores and the predicted sentiments###
    sentiments_from_scores, scores = separate_sentiments(loaded_full_dataset, n)
    predicted_sentiments = [entry["sentiment"] for entry in loaded_llm_predictions]

    ### Ensuginr everything works as intended: lengths match, and there is not errors in the sentiments###
    if len(sentiments_from_scores) != len(scores):
        raise ValueError(f"The length of sentiments ({len(sentiments_from_scores)}) does not match the length of scores ({len(scores)})")
    elif len(sentiments_from_scores) != len(predicted_sentiments):
        raise ValueError(f"The length of sentiments ({len(sentiments_from_scores)}) does not match the length of the predictions ({len(predicted_sentiments)})")
    
    check_sentiment_errors(sentiments_from_scores)
    check_sentiment_errors(predicted_sentiments)
    
        

    
if __name__ == "__main__":
    main()