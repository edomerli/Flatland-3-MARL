import torch
import numpy as np

def dict_to_tensor(dictionary):
    return torch.Tensor(np.array([dictionary[key] for key in dictionary.keys()]))

def tensor_to_dict(tensor):
    return {i: tensor[i].item() for i in range(tensor.size(0))}