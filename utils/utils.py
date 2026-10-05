import random

def shuffle_data(data, n= 42):
    '''
    Randomly shuffles the data.
    Seed is 42 unless otherwise specified.
    '''
    random.seed(n)
    random.shuffle(data)
    return data

def process_test_dataset(test_dataset):
    test_data = []
    for dct in test_dataset:
        for k, v in dct.items():
            if "reason_" in k:
                test_data.append(v)
    
    #random.shuffle(test_data)
    return test_data