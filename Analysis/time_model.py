import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils import data
import sys

new_dtype = torch.float32

def add_prefix(lst, prefix="X_new"):
    """
    Add prefix to list of file names
    @param lst: list of file names
    @param prefix: prefix to add
    @return: list of file names with prefix
    """
    return [prefix + "_" + str(i) + ".npy" for i in lst]


def add_prefix(lst, prefix="X_new"):
    return [prefix + "_" + str(i) + ".npy" for i in lst]

def CustomSequenceTiming(data_dir, file_idx, prefix="X_new"):
    Xnames = add_prefix(file_idx)
    X = None
    for x in (Xnames):
        if X is None:
            X = np.load(os.path.join(data_dir, x))[::100]
        else:
            X = np.concatenate((X, np.load(os.path.join(data_dir, x))[::100]))
    return X


def time_prediction(model, model_param_path=None, test_dataset=None, batch_size=200, use_gpu=True, data_parallel=False, verbose=1):
    if test_dataset is None:
        print("Please pass in the dataset using `CustomSequenceTiming()` function.")
        sys.exit(0)
        
    if model_param_path is not None:
        params = torch.load(model_param_path)
        if type(params) == dict and "model_state_dict" in params:
            params = params["model_state_dict"]
        try:
            if use_gpu:
                model.load_state_dict(params)
            else:
                model.load_state_dict(params, map_location="cpu")
                
        except:
            model = torch.nn.DataParallel(model)
            if use_gpu:
                model.load_state_dict(params)
            else:
                model.load_state_dict(params, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if use_gpu:
        if device == "cpu":
            Warning("GPU not available, using CPU instead.")
        if data_parallel:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            else:
                Warning("Data parallelism not available, using single GPU instead.")
        else:
            try:
                model = model.module()
            except:
                pass
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    model.to(device)
    model.eval()
    all_preds = []
    time1 = time.time()
    final_shape = None
    i = 0
    with torch.no_grad():
        for X_batch in test_dataloader:
            new_dtype = torch.float32
            X_batch = X_batch.to(new_dtype)
            if verbose:
                print(f"Predicting batch {i+1}/{len(test_dataloader)}")
            if use_gpu:
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                X_batch = X_batch.to(device)
            if final_shape is None:
                final_shape = X_batch.shape[-1]
            for _ in range(100):  # need to predict 100 times
                if not _ % 10 and verbose:
                    print("iter", _)
                pred = model(X_batch)
                X_batch = X_batch[:, 1:, :]  # pop first
                # add to last
                X_batch = torch.cat((X_batch, torch.reshape(pred, (-1, 1, final_shape))), 1)
            all_preds.append(pred.squeeze().cpu().numpy())
    time2 = time.time()
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, time2-time1
