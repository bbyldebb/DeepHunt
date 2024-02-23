'''
RCScorer: 
The root cause score of the current node = 
f(current node's reconstruction error, one-hop upstream node's reconstruction error, one-hop downstream node's reconstruction error)
'''
import dgl
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.layers import create_dataloader_fd, RankingLoss
import dgl.function as fn

def naive_scorer(cases, samples, loss_df, node_hash, window_size=10, pre=0.0, suc=-0.0):
    # print('window_size:', window_size, 'pre:', pre, 'suc', suc)
    test_df = cases.copy(deep=True)
    for case_id, case in cases.iterrows():
        start_ts = case['timestamp'] - (window_size*60/2)
        end_ts = case['timestamp'] + (window_size*60/2)
        # temporal dependency
        series_res = loss_df[(loss_df['timestamp']>=start_ts)&(loss_df['timestamp']<end_ts)].set_index('timestamp').mean()
        # upstream and downstream dependencies
        g = [sample for sample in samples if sample[0] >= start_ts and sample[0] < end_ts][-1][1]
        score_map = {node_hash[node]: series_res[node] for node in node_hash}
        final_score_map = {}
        for i in score_map:
            pre_nodes = g.successors(i).numpy().tolist()
            pre_nodes.remove(i) # remove self
            pre_score = np.max([score_map[p] for p in pre_nodes]) if len(pre_nodes) > 0 else 0
            suc_nodes = g.predecessors(i).numpy().tolist()
            suc_nodes.remove(i) # remove self
            suc_score = np.max([score_map[s] for s in suc_nodes]) if len(suc_nodes) > 0 else 0
            final_score_map[i] = score_map[i] + pre * pre_score + suc * suc_score
        ranks = sorted([(node, final_score_map[node_hash[node]]) for node in node_hash], key=lambda x: x[1], reverse=True)
        for i in range(len(ranks)):
            test_df.loc[case_id, f'Top{i+1}'] = '%s:%s' % ranks[i]
    return test_df

# Define the root cause scoring function model.
class RCScorer(nn.Module):
    def __init__(self, window_size, in_feats, out_feats):
        super(RCScorer, self).__init__()

        self.series_linear = nn.Linear(window_size, 1, bias=False)
        weights = torch.tensor([[1/window_size]*window_size])
        self.series_linear.weight.data.copy_(weights)

        self.fn = fn.copy_u('feat', 'max')
        self.graph_linear = nn.Linear(in_feats, out_feats, bias=False)
        weights = torch.tensor([0, -0])
        self.graph_linear.weight.data.copy_(weights)  #  initialize the weights of the FC

        self.softmax = nn.Softmax(dim=0)

    def aggregate_series(self, loss_series):
        # agg_feats = torch.mean(loss_series, dim=-1, keepdim=True)
        agg_feats = self.series_linear(loss_series)
        return agg_feats
    
    # aggregate the loss information from upstream and downstream nodes
    def aggregate_graph(self, g, h):
        with g.local_scope():
            # eliminate self-loop and do not include current node information when aggregating upstream and downstream node information.
            g.ndata['feat'] = h
            gs = g.remove_self_loop()
            # aggregate the features of predecessor nodes.
            gs.update_all(self.fn, fn.max('max', 'pre_feat'))
            # aggregate the features of successor nodes.
            gs_rev = gs.reverse()
            gs_rev.update_all(self.fn, fn.max('max', 'suc_max'))
            # Concatenate the features of each node.
            agg_feats = torch.cat([gs.ndata['pre_feat'], gs_rev.ndata['suc_max']], dim=1)
        return agg_feats
    
    def forward(self, g, loss_series):
        series_h = self.aggregate_series(loss_series)
        graph_h = self.graph_linear(self.aggregate_graph(g, series_h))
        output = self.softmax(series_h + graph_h)
        return output

class FdModel(nn.Module):
    def __init__(self, gae, window_size, in_feats, out_feats):
        super(FdModel, self).__init__()
        self.gae = gae
        self.mse = nn.MSELoss(reduction='none')
        self.scorer = RCScorer(window_size, in_feats, out_feats)
    
    def forward(self, g, feats):
        loss_series = torch.mean(self.mse(self.gae(g, feats), feats), dim=-1)
        h = self.scorer(g, loss_series)
        return h

# get labels for training model with feedback
def get_feedback_samples(cases, samples, node_hash, window_size=10, batch_size=1):
    fd_samples, labels = [], []
    for case_id, case in cases.reset_index().iterrows():
        start_ts = case['timestamp'] - (window_size*60)/2
        end_ts = case['timestamp'] + (window_size*60)/2
        label = torch.zeros(samples[0][1].num_nodes())
        for node in node_hash:
            if node.startswith(case['cmdb_id']):
                label[node_hash[node]] = 1.0
        labels.append(label)
        series = []
        for sample in samples:
            if sample[0] < start_ts:
                continue
            elif sample[0] >= start_ts and sample[0] < end_ts:
                series.append(sample)
            elif sample[0] >= end_ts:
                break
        if len(series) < window_size:
            series = [series[0]] * (window_size-len(series)) + series
        elif len(series) > window_size:
            series = series[-window_size:]
        fd_samples.append(series)
    return create_dataloader_fd(fd_samples, labels, batch_size, shuffle=False)

def feedback(model, cases, samples, node_hash, config):
    epochs = config['epochs']
    print('Freeze the GAE parameters: %s' % config['frozen'])
    if config['frozen']:
        for param in model.parameters():
            param.requires_grad = False
    model.mask_rate = config['mask_rate']
    fd_model = FdModel(model, config['window_size'], 2, 1)
    criterion = RankingLoss(rank_range=samples[0][1].num_nodes())
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, fd_model.parameters()), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    dataloader = get_feedback_samples(cases, samples, node_hash, window_size=config['window_size'], batch_size=config['batch_size'])
    
    best_state_dict= fd_model.state_dict()
    PATIENCE = 50
    early_stop_threshold = 1e-6
    prev_loss = np.inf
    stop_count = 0
    
    for epoch in tqdm(range(epochs)):
        running_loss = []
        for batched_graphs,  batched_feats, batched_labels in dataloader:
            scores = fd_model(batched_graphs, batched_feats)
            loss = criterion(scores.view(-1), batched_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss.append(loss.item())
        epoch_loss = np.mean(running_loss)
        if prev_loss - epoch_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                print('Early stopping')
                fd_model.load_state_dict(best_state_dict)
                break
        else:
            best_state_dict= fd_model.state_dict()
            stop_count = 0
            prev_loss = epoch_loss
        if epoch % 50 == 0:
            print(f'epoch {epoch} loss: ', np.mean(running_loss))
        scheduler.step(np.mean(running_loss))
    return fd_model
