import torch
import numpy as np

def data_aug(g, features, mask_rate):
    # Randomly mask a certain number of vertices.
    mask = torch.rand(g.number_of_nodes()) < mask_rate
    # Randomly mask a certain number of feature dimensions.
    z = features * torch.from_numpy(np.random.choice([0, 1], size=features.shape[-1], p=[mask_rate, 1-mask_rate])).float()
    z = torch.where(mask.unsqueeze(-1), torch.zeros_like(z), z)
    return g, z