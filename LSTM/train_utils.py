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


def add_prefix(lst, prefix="X"):
    """
    Add prefix to list of file names
    @param lst: list of file names
    @param prefix: prefix to add
    @return: list of file names with prefix
    """
    return [prefix + "_" + str(i) + ".npy" for i in lst]


class CustomSequence(data.Dataset):
    def __init__(self, data_dir, file_idx, file_batch_size, model_batch_size, test_mode=False, ):
        """
        Custom PyTorch dataset for loading data
        @param data_dir: directory containing data
        @param file_idx: list of file indices to load
        @param file_batch_size: number of files to load at once
        @param model_batch_size: number of samples to load at once to feed into model
        @param test_mode: whether to load data for testing
        """
        self.Xnames = add_prefix(file_idx)
        self.ynames = add_prefix(file_idx, "y")
        self.file_batch_size = file_batch_size
        self.model_batch_size = model_batch_size
        self.test_mode = test_mode
        self.data_dir = data_dir

    def __len__(self):
        return int(np.ceil(len(self.Xnames) / float(self.file_batch_size)))

    def __getitem__(self, idx):
        batch_x = self.Xnames[idx *
                              self.file_batch_size:(idx + 1) * self.file_batch_size]
        batch_y = self.ynames[idx *
                              self.file_batch_size:(idx + 1) * self.file_batch_size]

        data = []
        labels = []

        for x, y in zip(batch_x, batch_y):
            if not self.test_mode:
                temp_x = np.load(os.path.join(self.data_dir, x))
                temp_y = np.load(os.path.join(self.data_dir, y))
            else:
                temp_x = np.load(os.path.join(self.data_dir, x))[::100]
                temp_y = np.load(os.path.join(self.data_dir, y))[99:][::100]

            data.extend(temp_x)
            labels.extend(temp_y)

        for i in range(0, len(data), self.model_batch_size):
            data_batch = data[i:i + self.model_batch_size]
            labels_batch = labels[i:i + self.model_batch_size]

            data_tensor = torch.from_numpy(np.array(data_batch))
            label_tensor = torch.from_numpy(np.array(labels_batch))

            yield data_tensor, label_tensor


def train(model, train_dataset, num_epochs=10, val_dataset=None, use_gpu=True, data_parallel=True, out_dir=".", model_name="model", verbose=1, save_checkpoints=True, custom_loss=None):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    if use_gpu:
        if device == "cpu":
            Warning("GPU not available, using CPU instead.")
        if data_parallel:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
                print("Using", torch.cuda.device_count(), "GPUs!")
            else:
                Warning("Data parallelism not available, using single GPU instead.")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_losses = []
    if val_dataset is not None:
        val_losses = []

    # Train
    for epoch in range(num_epochs):
        if verbose:
            print("Epoch", epoch + 1)
            
        model.train()
        train_loss = 0
        train_len = 0
        for i in range(len(train_dataset)):
            sample_generator = train_dataset[i]
            for X, y in sample_generator:
                train_len += X.size(0)
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X.size(0)
        train_loss /= train_len
        checkpoint_path = os.path.join(out_dir, f"{model_name}_epoch_{epoch+1}.pth")
        if save_checkpoints:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)

        # Validation
        if val_dataset is not None:
            model.eval()
            val_loss = 0
            val_len = 0
            with torch.no_grad():
                for i in range(len(val_dataset)):
                    sample_generator = val_dataset[i]
                    for X, y in sample_generator:
                        X, y = X.to(device), y.to(device)
                        val_len += X.size(0)
                        # X, y = X.cuda(), y.cuda()
                        pred = model(X)
                        loss = criterion(pred, y)
                        val_loss += loss.item() * X.size(0)
            val_loss /= val_len


        train_losses.append(train_loss)
        np.save(os.path.join(out_dir, "train_losses.npy"), np.array(train_losses))
        if val_dataset is not None:
            val_losses.append(val_loss)
            np.save(os.path.join(out_dir, "val_losses.npy"), np.array(val_losses))

        if verbose:
            if val_dataset is not None:
                print(f'Epoch {epoch + 1}: Train Loss={train_loss:.18f}, Val Loss={val_loss:.18f}')
            else:
                print(f'Epoch {epoch + 1}: Train Loss={train_loss:.18f}')

    torch.save(model.state_dict(), os.path.join(out_dir, f'{model_name}.pth'))
    return model, train_losses, val_losses


def predict(model, model_param_path=None, test_dataset=None, use_gpu=True, data_parallel=False, output_dir=".", output_name="all_preds.npy", verbose=1):
    if model_param_path is not None:
        try:
            model.load_state_dict(torch.load(model_param_path))
        except:
            model = model.data_parallel()
            model.load_state_dict(torch.load(model_param_path))

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
    model.to(device)

    model.eval()
    all_preds = []
    final_shape = None
    with torch.no_grad():
        for i in range(len(test_dataset)):
            if verbose:
                print(f"Predicting batch {i+1}/{len(test_dataset)}")
            for X, y in test_dataset:
                X, y = X.to(device), y.to(device)
                if final_shape is None:
                    final_shape = y.shape[1]
                pred = model(X)
                for _ in range(100): # need to predict 100 times
                    pred = model(X)
                    X = X[:, 1:, :] # pop first
                    X = torch.cat((X, torch.reshape(pred, (-1, 1, final_shape))), 1) # add to last
                all_preds.append(pred.squeeze().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    np.save(os.path.join(output_dir, f"{output_name}.npy"), all_preds)
