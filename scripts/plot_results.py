import sys
import os
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def align_predictions_and_scores(predicted_sentiments, scores):
    """
    Align the predicted sentiment with the corresponding score
    """

    sent_to_scores = {}

    for sent, score in zip(predicted_sentiments, scores):
            if sent not in sent_to_scores:
                sent_to_scores[sent] = [score]
            else: 
                sent_to_scores[sent].append(score)

    return sent_to_scores

def create_plot_df(llm_to_scores, sent:str):
    '''
    Create an appropraite dataframe for the violin plots.
    Takes a dictionry (llm_to_scores) as well as a sentiment
    Either "Positive", "Negative", "Neutral".

    First, creates a plot_df dictionary, which has the following structure:
    Positive dictionary =
    {llm_name_1: [1, 0.9, 1],
    llm_name_2: [1, 0.8, 0.8]}

    It creates one dictionary like this given a sent.

    Returns
    -------
    df: dataframe
        a dataframe containins two columns: the llm name and the corresponding scores
    '''

    plot_df = {}

    for k, v in llm_to_scores.items():
        if len(v[sent]) > 1:
            plot_df[k]= v[sent]
    
    df_dct = {}
    df_dct["LLM"] = [k for k,v in plot_df.items() for value in v] #makes a column contianing the LLM name
    df_dct["Scores"] = [v for value_list in plot_df.values() for v in value_list] #flatten the list of values
    df = pd.DataFrame(df_dct)

    return df

def plot_results(df, name):
    #print(df)
    #sns.set_theme(font = "sans-serif", style = "ticks", palette="pastel")
    f = plt.figure()
    f.set_figwidth(3)
    f.set_figheight(3)
    sns.set_palette(["#F7C9AF", "#C7ECBA", "#E9ADE2"])
    ax = sns.violinplot(data=df, x = "LLM", y = "Scores", bw_adjust=.5, cut=1, linewidth=1, hue = "LLM")
    #ax.set_xticks(["Llama", "Qwen", "Gemma"])
    ax.set_xticklabels(["Llama", "Qwen", "Gemma"])
    plt.xticks(rotation=0, rotation_mode = "default", fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.ylim(top=1.05)
    plt.ylim(bottom=-0.09)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    plt.savefig("plots/" + name + ".svg", dpi=300, bbox_inches="tight")
    plt.clf()

def main():

    parser = argparse.ArgumentParser(
        description="Select examples for 1-shot learning"
    )

    parser.add_argument(
        "--llm_predictions",
        required=False,
        help="The json file containing the llm predictions"
    )     

    parser.add_argument(
        "--full_dataset",
        required=False,
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

    llm_predictions = [
        "results/classification/magistral:24b_llama3.1:latest_Nonesamples_verbose_full1_swappingFalse.json",
        "results/classification/magistral:24b_qwen2.5:32b_Nonesamples_verbose_full1_swappingFalse.json",
        "results/classification/magistral:24b_gemma3:12b_Nonesamples_verbose_full1_swappingFalse.json"
    ]

    full_datasets = [
        "data/full_datasets/llama3.1:latest_Nonesamples_verbose_full1_swappingFalse.json",
        "data/full_datasets/qwen2.5:32b_Nonesamples_verbose_full1_swappingFalse.json",
        "data/full_datasets/gemma3:12b_Nonesamples_verbose_full1_swappingFalse.json"
    ]

    llm_to_scores = {}

    for full_dataset, llm_prediction in zip(full_datasets, llm_predictions):

        ###Ensuring the right datasets are being compared###

        llm_name = full_dataset.split("_")[1].split("/")[-1]
        llm_predictions_name = llm_prediction.split("_")[1]

        if llm_name != llm_predictions_name:
            print("ERROR. The wrong test dataset has been chosen.")

        ### Loading the data ###
        loaded_llm_prediction = is_json_or_jsonl(llm_prediction)
        loaded_full_dataset = is_json_or_jsonl(full_dataset)

        ###Extracting the sentiments from the scores and the predicted sentiments###
        sentiments_from_scores, scores = separate_sentiments(loaded_full_dataset, n)
        predicted_sentiments = [entry["sentiment"] for entry in loaded_llm_prediction]

        ### Ensuginr everything works as intended: lengths match, and there is not errors in the sentiments###
        if len(sentiments_from_scores) != len(scores):
            raise ValueError(f"The length of sentiments ({len(sentiments_from_scores)}) does not match the length of scores ({len(scores)})")
        elif len(scores) != len(predicted_sentiments):
            raise ValueError(f"The length of scores ({len(scores)}) does not match the length of the predictions ({len(predicted_sentiments)})")
        
        check_sentiment_errors(sentiments_from_scores)
        check_sentiment_errors(predicted_sentiments)
            
        sent_to_scores = align_predictions_and_scores(predicted_sentiments, scores)
        llm_to_scores[llm_name] = sent_to_scores    
    
    sent = ["Positive", "Neutral", "Negative"]

    for s in sent:
        df = create_plot_df(llm_to_scores, s)
        plot_results(df, s)
       
    
if __name__ == "__main__":
    main()