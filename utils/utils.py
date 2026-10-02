import random

def shuffle_data(data, n= 42):
    '''
    Randomly shuffles the data.
    Seed is 42 unless otherwise specified.
    '''
    random.seed(n)
    random.shuffle(data)
    return data