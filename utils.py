import os
import numpy as np
import pandas as pd
import pickle
import torch
import shutil
from datetime import datetime

def printf(text):
    print('%s ---- %s' % (str(datetime.today())[:19], text))

def pathjoin(*args):
    return os.path.join(*args).replace('\\', '/')

def pathmake(dir_name, clean=False):
    if os.path.isdir(dir_name):
        if clean:
            shutil.rmtree(dir_name)
    else:
        os.makedirs(dir_name)

def list_all_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def remove_files(dir_name, pattern='temp'):
    for f in list_all_files(dir_name):
        if pattern in f:
            os.remove(f)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def random_spaced(low, high, delta, n, size=None):
    if n == 0:
        return []
    empty_space = high - low - (n-1)*delta
    if empty_space < 0:
        return []
    if size is None:
        u = np.random.rand(n)
    else:
        u = np.random.rand(size, n)
    x = empty_space * np.sort(u, axis=-1)
    return (low + x + delta * np.arange(n)).astype(int)

def intersects(a, b):
    return a[0] < b[1] and b[0] < a[1]

def list_diff(l1, l2):
    return (np.array(l1) - np.array(l2)).tolist()

def list_fill(l, n):
    r = n - len(l)
    return l if r == 0 else l + [1] * r

def argmedian(arr):
    sorted_indices = np.argsort(arr)
    n = len(arr)
    median_index = sorted_indices[n // 2]
    return median_index