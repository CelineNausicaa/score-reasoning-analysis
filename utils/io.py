from pathlib import Path
import json
from json import JSONDecodeError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR = PROJECT_ROOT / "data" 
# RESULTS_DIR = PROJECT_ROOT / "results"

from .utils import shuffle_data

def load_json(file_name, n = None, shuffle = False): #USED
    '''
    Load a JSON file. If n is provided, only gets the first n elements.
    
    Parameters
    ---------
    file_name: string
        path to the json file
    
    n: int
        load n samples
    
    shuffle: float
        if True, the dataset is shuffled
    
    Returns
    -------
    data: file object
        the loaded json file
    '''

    with open(file_name, 'r') as file:
        data = json.load(file)
    
    if shuffle:
        shuffle_data(data)
    
    if n:
        subset_data = data[:n]
        return subset_data

    return data

def output_json(data, outdir, shuffling = False, seed = 42): #USED
    '''
    Outputs the data to a json file.
    
    Parameters
    ----------
    data: a list
        a list of dictionaries
    outdir: string
        a string indicating the name of the output directory
    '''

    if shuffling:
        rng = random.Random(seed)
        rng.shuffle(data)
    
    print(f"Exporting to JSON.\nOutput directory: {outdir}")
    with open(outdir, 'w') as f:
        json.dump(data, f, indent=4)

def append_jsonl(entry, outdir):
    '''
    Write to a JSONL file little by little, rather than getting everything all at once.
    '''
    with open(outdir, "a", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")

def load_jsonl(file_name):
    '''
    Load a JSONL file.
    Takes a subset n for each of the file, so that a balanced sample is loaded.
    
    Parameters
    ----------
    file_name: string
        path to the jsonl file
    ´
    Returns
    -------
    data: file object
        the loaded jsonl file
    '''

    with open(file_name, "r") as file:
        data = [json.loads(line) for line in file if line.strip()]

    return data

def is_json_or_jsonl(file):
    try:
        data = load_json(file)
    except JSONDecodeError as e:
        data = load_jsonl(file)
    return data