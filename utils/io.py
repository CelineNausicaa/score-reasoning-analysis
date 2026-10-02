from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" 
RESULTS_DIR = PROJECT_ROOT / "results"

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