import time
import json
import pandas as pd
import os
import pickle
import numpy as np

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def date2ts(date_str):
    return int(time.mktime(time.strptime(date_str, '%Y-%m-%d')))

def load_feats(f_name, filedir):
    return pd.read_csv(os.path.join(filedir, f_name))

def load_samples(filedir):
    with open(os.path.join(filedir, 'test_samples.pkl'), 'rb') as f:
        test_samples = pickle.load(f)
    with open(os.path.join(filedir, 'train_samples.pkl'), 'rb') as f:
        train_samples = pickle.load(f)
    return train_samples, test_samples

def load_init(filedir):
    try:
        with open(os.path.join(filedir, 'call_edges.pkl'), 'rb') as f:
            call_info = pickle.load(f)
    except:
        call_info = []
    try:
        with open(os.path.join(filedir, 'host_edges.pkl'), 'rb') as f:
            host_pairs = pickle.load(f)
    except:
        host_pairs = []
    with open(os.path.join(filedir, 'node_hash.pkl'), 'rb') as f:
        node_hash = pickle.load(f)
    host_pods, host_nodes = [], []
    for pair in host_pairs:
        host_pods.append(node_hash[pair.split('.')[1]])
        host_nodes.append(node_hash[pair.split('.')[0]])
    callers, callees = [], []
    for caller, callee in sorted(call_info):
        if caller == '#':
            continue
        callers.append(node_hash[caller])
        callees.append(node_hash[callee])
    vecs_i = np.concatenate([host_nodes, host_pods, list(node_hash.keys()), callers])
    vecs_j = np.concatenate([host_pods, host_nodes, list(node_hash.keys()), callees])
    vecs_i, vecs_j = zip(*set(zip(vecs_i, vecs_j)))
    return node_hash, vecs_i, vecs_j
