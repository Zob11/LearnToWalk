import torch 
from numpy import dot
from numpy.linalg import norm
import torch.nn as nn
import numpy as np
import random
from torch.nn import Module
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import List, Union, Dict
from collections import defaultdict
import scipy.signal as ss
from sklearn.utils import shuffle
import pickle
import os
from tcn import TemporalConvNet
import copy
from tqdm import tqdm
from torch.optim import SGD, Adam
from torch.nn import MSELoss
import argparse
from train_utils import Replay_Data, TaskBalancedDataset, get_device, \
    make_train_val_dataloader

#%%declarations and functions
seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g_seed = torch.Generator()
g_seed.manual_seed(seed)
#training specific declaration   
device = get_device()

#%% Model architecture

class MultiGaitAdapter(Module):
    
    def __init__(self, in_features, initial_out_features=1, n_heads=None, head_ids=None):
        
        super().__init__()
        self.in_features = in_features
        self.starting_out_features = initial_out_features
        self.out_channels = 100
        
        self.baselayers = TemporalConvNet(
            num_inputs=in_features, 
            num_channels=(self.out_channels,
                          self.out_channels*2), 
            kernel_size=5, 
            dropout=0.2)
        self.prediction_heads = torch.nn.ModuleDict()
        if head_ids is None:
            assert n_heads is not None
            head_ids = np.arange(1, n_heads+1)
        first_head = torch.nn.Sequential(nn.Linear(self.out_channels*2, self.out_channels),
                                         nn.Tanh(),
                                         nn.Linear(self.out_channels,self.starting_out_features)
                                         )
        for h in head_ids:
            self.prediction_heads[str(h)] = copy.deepcopy(first_head)


    def forward(self, x, task_labels):
        out1 = x
        out2 = self.baselayers(out1.float())[:, :, -1]
                
        unique_tasks = np.unique(task_labels)

        out = None
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = out2[task_mask]
            out_task = self.prediction_heads[str(task)](x_task)

            if out is None:
                out = torch.empty(x.shape[0], *out_task.shape[1:],
                                  device=out_task.device)
            out[task_mask] = out_task
        return out
    
#train and validation loop
def multihead_training(model, nb_epochs, train_dataloader, criterion, optimizer, 
             val_loader=None, model_save_path='best_model.pth'):
    min_valid_loss = np.inf
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        for X, y, t in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):            
            # Transfer Data to GPU if available
            if torch.cuda.is_available():
                X, y = X.cuda(), y.cuda()       
            # zero the parameter gradients
            optimizer.zero_grad()    
            # forward + backward + optimize
            y_pred = model(X, t)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()    
            # Calculate Loss
            train_loss += loss.item()
            
        if val_loader is not None:
            valid_loss = 0.0
            model.eval()     
            for X_val, y_val, t_val in val_loader:
                # Transfer Data to GPU if available
                if torch.cuda.is_available():
                    X_val, y_val = X_val.cuda(), y_val.cuda()             
                # Forward Pass
                target = model(X_val, t_val)
                # Find the Loss
                loss = criterion(target, y_val)
                # Calculate Loss
                valid_loss += loss.item() 
            valid_loss = valid_loss/len(val_loader)
            print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss}')
            if min_valid_loss > valid_loss:
                min_valid_loss = valid_loss                
                # Saving State Dict
                if model_save_path is not None:
                    print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                    torch.save(model.state_dict(), model_save_path)
        else:
            print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)}')
    
    print('Finished Training')
    return train_loss / len(train_dataloader)

def multihead_validation(net, test_iters, criterion):
    net.eval()
    valid_loss = 0.0
    actual = []
    predicted = []
    tasks = []
    for X, y, t in tqdm(test_iters):
        if torch.cuda.is_available():
            X, y = X.cuda(), y.cuda()
        y_pred = net(X, t)
        actual.append(y.detach().numpy())
        predicted.append(y_pred.detach().numpy())
        tasks.append(t.detach().numpy())
        loss = criterion(y_pred, y)
        valid_loss += loss.item()
    return valid_loss/len(test_iters), np.concatenate(actual), \
        np.concatenate(predicted), np.concatenate(tasks)

#%% get parameters
my_parser = argparse.ArgumentParser()

my_parser.add_argument('--dataset', action='store', type=str, default='my_data', required=False)
my_parser.add_argument('--backward', action='store', type=bool, default=True, required=False)
my_parser.add_argument('--noise', action='store', type=float, default=0.0, required=False)
args = my_parser.parse_args()
dataset =  args.dataset
ignore_backward = not args.backward
inp_noise_std = args.noise

if dataset == 'my_data':
    
    my_parser.add_argument(
        '--input', action='store', type=str, 
        default='../data/my_data', 
        required=False)
    my_parser.add_argument(
        '--output', action='store', type=str, 
        default='../results/mutihead_joint', 
        required=False)
    
    subjects = ['subject1']
    
    task_ids = [1, 2, 3]
    
    inputs = ['input_1', 'input_2']
    
    target = 'output'
    
    task_col_name = 'task_id'
    
    normalize_inp=False
    normalize_tar=False
    lowpassfilter=True

args = my_parser.parse_args()

results_folder = args.output
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

#%% data preparation
for subject_name in subjects:
    print(subject_name)
    model = MultiGaitAdapter(in_features = len(inputs), head_ids=task_ids)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = MSELoss(reduction="sum")
    batch_size = 100
    val_size = None
    nb_replay = np.inf
    nb_epochs = 25
    model_save_path=os.path.join(
        results_folder, f'{subject_name}_best_model.pth')
    
    df_results = pd.DataFrame()
    
    test_iters = []
    for j, task_id in enumerate(task_ids):
        subject_testset = Replay_Data(f"{args.input}/{subject_name}_test.csv", 
                                      task_id, inputs, target, task_col_name, 
                                      normalize_inp=normalize_inp, 
                                      normalize_tar=normalize_tar, 
                                      lowpassfilter=lowpassfilter, 
                                      ignore_backward=ignore_backward,
                                      inp_noise_std = inp_noise_std, 
                                      time_delay=True, shuffle_data=False)
        test_iters.append(subject_testset)
    testsets_merged = TaskBalancedDataset(test_iters, False)
    test_dataloader = DataLoader(testsets_merged, shuffle=False, batch_size=100)
    
    # training
    train_iters = []
    for i, train_task_id in enumerate(task_ids):
        subject_trainset = Replay_Data(
            f"{args.input}/{subject_name}_train.csv", 
            train_task_id, inputs, target, task_col_name,
            normalize_inp=normalize_inp, 
            normalize_tar=normalize_tar, 
            lowpassfilter=lowpassfilter, 
            ignore_backward=ignore_backward,
            time_delay=True,)
        train_iters.append(subject_trainset)
                    
    trainset = TaskBalancedDataset(train_iters, False)
    train_dataloader = DataLoader(trainset, shuffle=True, batch_size=100)
        
    train_loss = multihead_training(
        model, nb_epochs, train_dataloader, criterion, optimizer, 
        val_size, model_save_path)
    valid_loss, actual, predicted, task_labels = multihead_validation(
        model, test_dataloader, criterion)
    
    # save model
    torch.save(model.state_dict(), f'{results_folder}/{subject_name}_model.pth')
    
    #%% collect results
    model.load_state_dict(torch.load(f'{results_folder}/{subject_name}_model.pth'))
    valid_loss, actual, predicted, task_labels = multihead_validation(
        model, test_dataloader, criterion)
    print('\n')
    # validation
    import matplotlib.pyplot as plt
    plt.figure()
    true_predicted = {}
    for test_task_id in task_ids:
        r2_valid = r2_score(
            actual[task_labels==test_task_id], 
            predicted[task_labels==test_task_id])
        rmse_valid = np.sqrt(mean_squared_error(
            actual[task_labels==test_task_id], 
            predicted[task_labels==test_task_id]))
        print(test_task_id, r2_valid)
        plt.scatter(
            actual[task_labels==test_task_id], 
            predicted[task_labels==test_task_id], 
            s=1, label=f'task {test_task_id}: R2={r2_valid:0.2f}')
        df_results = df_results.append(
            {'rmse_valid': rmse_valid, 
             'test_task_id': test_task_id, 
             'r2_valid': r2_valid}, 
            ignore_index=True)
        true_predicted[test_task_id] = {}
        true_predicted[test_task_id]['y_true'] = \
            actual[task_labels==test_task_id]
        true_predicted[test_task_id]['y_pred'] = \
            predicted[task_labels==test_task_id]
            
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), 'k', lw=2)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{results_folder}/{subject_name}.png', dpi=300)
    
    # save results
    df_results.to_csv(f'{results_folder}/results_{subject_name}.csv')
    
    # save results
    with open(f'{results_folder}/{subject_name}_true_vs_predicted.pickle', 'wb') as handle:
        pickle.dump(true_predicted, handle, protocol=pickle.HIGHEST_PROTOCOL)