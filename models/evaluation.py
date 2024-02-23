import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from utils.public_functions import load_init
from models.layers import ModalLoss
from models.rc_scorer import naive_scorer, feedback, get_feedback_samples

# test: output the loss for each node
def test(model, samples, node_hash, feat_span):
    mse = nn.MSELoss()
    # modal_loss = ModalLoss(feat_span)
    with torch.no_grad():
        res = {node: [] for node in node_hash}
        for ts, g, feats in samples:
            outputs = model.transform(g, feats)
            for node in node_hash:
                loss = mse(outputs[node_hash[node]], feats[node_hash[node]])
                # loss = modal_loss.compute(outputs[node_hash[node]], feats[node_hash[node]])
                res[node].append((ts, loss.item()))
    loss_df = pd.concat([pd.DataFrame(res[node], columns=['timestamp', node]).set_index('timestamp') for node in res], axis=1).reset_index()
    return loss_df

def fd_test(test_cases, fd_model, samples, node_hash, window_size):
    fd_test_df = test_cases.copy(deep=True).reset_index(drop=True)
    dataloader = iter(get_feedback_samples(test_cases, samples, node_hash, window_size, batch_size=1))
    for case_id in range(len(fd_test_df)):
        batched_graphs,  batched_feats, _ = next(dataloader)
        scores = fd_model(batched_graphs, batched_feats).tolist()

        final_score_map = {i: scores[i] for i in range(len(scores))}
        ranks = sorted([(node, final_score_map[node_hash[node]]) for node in node_hash], key=lambda x: x[1], reverse=True)
        for i in range(len(ranks)):
            fd_test_df.loc[case_id, f'Top{i+1}'] = '%s:%s' % ranks[i]
    return fd_test_df

# train the feedback model and evaluate the model
def get_eval_df(model, cases, samples, config):
    res_dict = dict()
    node_hash, _, _ = load_init(config['path']['graph_dir'])
    fd_num = config['feedback']['sample_num']
    test_index = -int(len(cases) * 0.7) # split the test set
    if isinstance(fd_num, int):
        fd_cases, test_cases = cases.iloc[: fd_num], cases.iloc[test_index: ]
    elif isinstance(fd_num, float) and (0 < fd_num < 1):
        n_cases = len(cases)
        split_pos = int(n_cases * fd_num)
        fd_cases, test_cases = cases.iloc[: split_pos], cases.iloc[test_index: ]
    else:
        raise Exception('invalid sample_num')
    print('feedback sample nums: %s; test sample nums: %s' % (len(fd_cases), len(test_cases)))
    
    print('Using the naive model / the model with default scorer for evaluation.')
    loss_df = test(model, samples, node_hash, config['model_param']['feat_span'])
    test_df = naive_scorer(test_cases, samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    res_all = evaluation(test_df, 5)
    test_df = naive_scorer(cases, samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    res_dict['naive_res_all'] = res_all.tolist()
    print('res_all: ', res_all)

    print('Using the model with feedback for evaluation.')
    fd_model = feedback(model, fd_cases, samples, node_hash, config['feedback'])
    fd_test_df = fd_test(test_cases, fd_model, samples, node_hash, config['feedback']['window_size'])
    res_all = evaluation(fd_test_df, 5)
    res_dict['fd_res_all'] = res_all.tolist()
    print('res_all: ', res_all)

    return fd_model, test_df, fd_test_df, res_dict

# calculate the TopK
def evaluation(cases, k=5):
    topks = np.zeros(k)
    for _, case in cases.iterrows():
        for i in range(k):
            if case['cmdb_id'] in case[f'Top{i+1}']:
                topks[i: ] += 1
                break
    return np.round(topks / len(cases), 4)
