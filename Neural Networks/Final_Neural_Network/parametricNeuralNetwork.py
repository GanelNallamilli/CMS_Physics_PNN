# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:38:39 2024

@author: drpla
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:53:03 2024

@author: drpla
"""

"""
THIS IS REDUNDANT 

Do not use this for anything, just holds a reference of my old code that Ganel wrote better
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import mplhep as hep
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import copy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def weightedBCELoss(input, target, weight):
  x, y, w = input, target, weight
  log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
  return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))

def BCELoss(input, target):
  x, y= input, target
  log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
  return torch.mean(-1 * (y*log(x) + (1-y)*log(1-x)))


def learning_rate_scheduler(epoch, lr_epoch, initial, epochlist, scheduler=None):
    newlr=lr_epoch
    if scheduler == 'Custom':
        if epoch%5 == 0 and epoch > 0:
            f=epochlist[epoch]/epochlist[0]
            newlrr=initial*f
            newlr=float(newlrr.item())
    #if scheduler == 'Linear':
        
    return newlr      

def replace_9(distribution):
    distribution.replace(-9, pd.NA, inplace=True)
    column_means = distribution.mean()
    distribution.fillna(column_means, inplace=True)
    return distribution
        




def read_dataframes(directory = '', signal_name = ''):
    #list of each bkgs for concatenation
    background_list=['DiPhoton', 
                     'TTGG', 
                     'TTGamma',
                     'VBFH_M125',
                     'VH_M125',
                     'WGamma',
                     'ZGamma',
                     'ggH_M125', 
                     'ttH_M125']
    
    #The features requiring exclusion of -9 values
    MinusNineBinning=['ditau_met_dPhi',
                      'ditau_deta',
                      'ditau_dR',
                      'ditau_dphi',
                      'ditau_pt',
                      'Diphoton_ditau_dphi',
                      'dilep_leadpho_mass',
                      'reco_MX_mgg',
                      'Diphoton_ditau_deta',
                      'Diphoton_sublead_lepton_deta',
                      'Diphoton_sublead_lepton_dR',
                      'LeadPhoton_ditau_dR',
                      'ditau_mass',
                      'jet_1_pt']

    df = pd.read_parquet(f'{directory}merged_nominal.parquet')
    with open(f'{directory}summary.json', "r") as f:
        proc_dict = json.load(f)["sample_id_map"]

    signal = df[df.process_id == proc_dict[f"{signal_name}"]]

    signal = replace_9(signal.copy())

    listforconc=[]
    for i in background_list:                              
        bkgg = df[df.process_id == proc_dict[i]]
        listforconc.append(bkgg)

    for i in range(len(listforconc)):
        listforconc[i] = replace_9(listforconc[i].copy())

    background = pd.concat(listforconc)


    signal['y']=np.ones(len(signal.index))
    background['y']=np.zeros(len(background.index))

    combine = pd.concat([signal,background])

    listforconc=[]
    add_to_test = ['GJets','TTJets']
    for i in add_to_test:                              
        bkgg = df[df.process_id == proc_dict[i]]
        listforconc.append(bkgg)

    add_to_test_df = pd.concat(listforconc)
    add_to_test_df['y']=np.zeros(len(add_to_test_df.index))

    return signal,background,combine,add_to_test_df


def paramDataframe():
    allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']
    for i in allmasses:
        signal_name = "GluGluToRadionToHHTo2G2Tau_M-"+i
        signal,background,combine,add_to_test_df = read_dataframes(directory = '', signal_name = '')
        signal[f'{i}'] = float(i)
        background[f'{i}'] = float(i) * len(signal) / len(background)
    return signal,background,combine,add_to_test_df
        
"""
def paramDataframe(directory = ''):
    #allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']
    allmasses=['260','270','280']
    df = pd.read_parquet(f'merged_nominal.parquet')

    signals=pd.DataFrame(columns=df.columns[:])
    backgrounds=pd.DataFrame(columns=df.columns[:])
    combines=pd.DataFrame(columns=df.columns[:])
    add_to_test_dfs = pd.DataFrame(columns=df.columns[:])
    for i in [signals,backgrounds,combines,add_to_test_dfs]:
        i['y']=None
        i['mass']=None
    for i in allmasses:
        signal_name = "GluGluToRadionToHHTo2G2Tau_M-"+i
        signal,background,combine,add_to_test_df = read_dataframes(directory = '', signal_name = signal_name)
        signal['mass'] = float(i)
        background['mass'] = float(i) * len(signal) / len(background)
        add_to_test_df['mass'] = float(i) * len(signal) / len(add_to_test_df)
        combine = pd.concat([signal,background])
        
        combines=pd.concat([combine,combines])#concatenating each mass into one dataframe
        signals=pd.concat([signal,signals])
        add_to_test_dfs=pd.concat([add_to_test_df,add_to_test_dfs])
        backgrounds=pd.concat([background,backgrounds])
    return signals,backgrounds,combines,add_to_test_dfs
"""         
def paramDataframe(directory = ''):
    #allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']
    allmasses=['260','270','280']
    df = pd.read_parquet(f'merged_nominal.parquet')

    signals=pd.DataFrame(columns=df.columns[:])
    backgrounds=pd.DataFrame(columns=df.columns[:])
    combines=pd.DataFrame(columns=df.columns[:])
    add_to_test_dfs = pd.DataFrame(columns=df.columns[:])
    for i in [signals,backgrounds,combines,add_to_test_dfs]:
        i['y']=None
        i['mass']=None
    for i in allmasses:
        signal_name = "GluGluToRadionToHHTo2G2Tau_M-"+i

        
        signal,background,combine,add_to_test_df = read_dataframes(directory = '', signal_name = signal_name)

        # Add a new column 'NewColumn' and assign a value to a subset of rows randomly
        
      


        signal['mass'] = float(i)
        num_random_rows = len(signal)
        background['mass'] = np.where(np.random.rand(len(background)) < (num_random_rows / len(background)), float(i), np.nan)
        #df['NewColumn'] = np.where(np.random.rand(len(df)) < (num_random_rows / len(df)), assigned_value, np.nan)


        background['mass'] = float(i) * len(signal) / len(background)
        add_to_test_df['mass'] = float(i) * len(signal) / len(add_to_test_df)
        combine = pd.concat([signal,background])
        
        combines=pd.concat([combine,combines])#concatenating each mass into one dataframe
        signals=pd.concat([signal,signals])
        add_to_test_dfs=pd.concat([add_to_test_df,add_to_test_dfs])
        backgrounds=pd.concat([background,backgrounds])
    return signals,backgrounds,combines,add_to_test_dfs

def getWeightedBatches(arrays, batch_size=None):
    weights = arrays[2]
    length = len(weights)
    indices = np.arange(length)


    offset = np.abs(np.min(weights)) if np.min(weights) < 0 else 0
    adjusted_weights = weights + offset
    sampling_prob = adjusted_weights / adjusted_weights.sum()


    #for _ in tqdm(range(0, length, batch_size)):
    for _ in range(0, length, batch_size):
        chosen_indices = np.random.choice(indices, size=batch_size, p=sampling_prob)
        arrays_batch = [torch.Tensor(array[chosen_indices]).to(device) for array in arrays]
        yield arrays_batch

def getBatches(arrays, batch_size=None):
    if len(arrays) != 3:
        raise Exception("'getBatchs' arrays should contain 3 variables: x,y,w (data, labels, weights)")
    
    length = len(arrays[0])

    #for i in tqdm(range(0, length, batch_size)):
    for i in range(0, length, batch_size):
        arrays_batch = [torch.Tensor(array[i:i+batch_size]).to(device) for array in arrays]
        yield arrays_batch

def basicModel(features, nodes):
    length = len(features)

    n_nodes = [length] + nodes + [1]
    layers = []
    for i in range(len(n_nodes)-1):
        if i == len(n_nodes)-2:
            layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))
            layers.append(nn.Sigmoid()) 
        else:
            layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))
            layers.append(nn.ReLU())  

    model = nn.Sequential(
        *layers
        )
    return model

def charModel(features, nodes):
    length = len(features)

    n_nodes = [length] + nodes + [1]
    layers = []
    for i in range(len(n_nodes)-1):
        if i == len(n_nodes)-2:
            layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))
            layers.append(nn.Sigmoid()) 
        else:
            layers.append(nn.Linear(n_nodes[i], n_nodes[i+1]))
            layers.append(nn.Dropout(0.05))
            layers.append(nn.ELU())  

    model = nn.Sequential(
        *layers
        )
    return model

def getTotalLoss_no_weight(model, loss_f, X, y, w, batch_size):
  total_loss = 0.0

  with torch.no_grad():
    for X_tensor, y_tensor, w_tensor in getWeightedBatches([X, y, w], batch_size):
      output = model(X_tensor)
      total_loss += loss_f(output, y_tensor.reshape(-1, 1)).detach().cpu()



    mean_loss = total_loss / len(X)

  return mean_loss

def getTotalLoss(model, loss_f, X, y, w, batch_size):
  total_loss = 0.0

  with torch.no_grad():
    for X_tensor, y_tensor, w_tensor in getBatches([X, y, w], batch_size):
      output = model(X_tensor)
      total_loss += loss_f(output, y_tensor.reshape(-1, 1), w_tensor.reshape(-1, 1)).detach().cpu()



    mean_loss = total_loss / len(X)

  return mean_loss

def getTrainTestSplit(combine_df,add_to_test_df = []):
    x_train, x_test, y_train, y_test = train_test_split(combine_df.drop(columns=['y']), combine_df['y'], test_size=(1/3), random_state=42)

    x_train['weight_central'][y_train==0] *= (x_train['weight_central'][y_train==1].sum() / x_train['weight_central'][y_train==0].sum())
    x_train['weight_central'] *= (len(x_train['weight_central']) / x_train['weight_central'].sum())

    x_test['weight_central'][y_test==0] *= (x_test['weight_central'][y_test==1].sum() / x_test['weight_central'][y_test==0].sum())
    x_test['weight_central'] *= (len(x_test['weight_central']) / x_test['weight_central'].sum())


    y_train = pd.DataFrame(y_train,columns=['y'])
    y_test = pd.DataFrame(y_test,columns=['y'])

    x_train['y'] = y_train
    x_test['y'] = y_test

    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    x_test = pd.concat([x_test, add_to_test_df], ignore_index=True)

    x_train = x_train.sample(frac=1).reset_index(drop=True)
    x_test = x_test.sample(frac=1).reset_index(drop=True)

    return x_train,x_test

def trainNetwork(train_df, test_df, features, lr,epoch = 200, outdir=None, save_models=False, batch_size = 1024,nodes = [5],model_type = 'basic',scheduler_type='None'):
    loss_f = lambda x, y, w: weightedBCELoss(x,y,w)

    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    X_train = train_df[features].to_numpy()
    X_test = test_df[features].to_numpy()
    y_train = train_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()
    w_train = train_df["weight_central"].to_numpy()
    w_test = test_df["weight_central"].to_numpy()

    if model_type == 'char':
        model = charModel(features,nodes).to(device)
    elif model_type == 'basic':
        model = basicModel(features,nodes).to(device)
    
    print(model)
    optimiser =  torch.optim.Adam(model.parameters(), lr= lr)
    epoch_loss_train = []
    epoch_loss_test = []
    models = []

    patience = 30
    best_loss = float('inf')
    patience_counter = 0
    learning_rate_epochs=[]
    print(">> Training...")
#    for i_epoch in tqdm(range(0,epoch)):
    for i_epoch in range(0,epoch):

       # print(f"Epoch {i_epoch}")
        total_loss = 0.0
        model.train()

        batch_gen = getBatches([X_train, y_train, w_train], batch_size = batch_size)

        for X_tensor, y_tensor, w_tensor in batch_gen:
            optimiser.zero_grad()
            output = model(X_tensor)
            loss = loss_f(output, y_tensor.reshape(-1, 1), w_tensor.reshape(-1, 1))
            total_loss += loss.detach().cpu()
            loss.backward()
            optimiser.step()

        model.eval()
        models.append(copy.deepcopy(model))

        epoch_loss_train.append(getTotalLoss(model, loss_f, X_train, y_train, w_train, batch_size))
        epoch_loss_test.append(getTotalLoss(model, loss_f, X_test, y_test, w_test, batch_size))

        
        for param_group in optimiser.param_groups:
            lr_epoch= param_group['lr']
            updated_lr = learning_rate_scheduler(epoch=i_epoch, lr_epoch=lr_epoch, initial=lr, epochlist=epoch_loss_train, scheduler=scheduler_type)
            param_group['lr'] = updated_lr
        learning_rate_epochs.append(updated_lr)
        
        
        if epoch_loss_test[-1] < best_loss:
            best_loss = epoch_loss_test[-1]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    #print(">> Training finished")
    model.eval()
    with torch.no_grad():
        output_score = model(torch.Tensor(X_test).to(device))
        output_score_train = model(torch.Tensor(X_train).to(device))

    return models,epoch_loss_train,epoch_loss_test,output_score,output_score_train, learning_rate_epochs

def trainNetwork_no_weights(train_df, test_df, features, lr,epoch = 200, outdir=None, save_models=False, batch_size = 1024,nodes = [5],model_type = 'basic',scheduler_type='None'):
    loss_f = lambda x, y: BCELoss(x,y)

    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    X_train = train_df[features].to_numpy()
    X_test = test_df[features].to_numpy()
    y_train = train_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()
    w_train = train_df["weight_central"].to_numpy()
    w_test = test_df["weight_central"].to_numpy()

    if model_type == 'char':
        model = charModel(features,nodes).to(device)
    elif model_type == 'basic':
        model = basicModel(features,nodes).to(device)
    
   # print(model)
    optimiser =  torch.optim.Adam(model.parameters(), lr= lr)
    epoch_loss_train = []
    epoch_loss_test = []
    models = []
    best_model = ""


    patience = 3
    best_loss = float('inf')
    patience_counter = 0
    learning_rate_epochs=[]
    best_epoch = 0
    #print(">> Training...")
   # for i_epoch in tqdm(range(0,epoch)):
    for i_epoch in range(0,epoch):
       # print(f"Epoch {i_epoch}")
        total_loss = 0.0
        model.train()
        if i_epoch%25 == 0:
            print(f'Epoch: {i_epoch}' )
        batch_gen = getWeightedBatches([X_train, y_train, w_train], batch_size = batch_size)

        for X_tensor, y_tensor, w_tensor in batch_gen:
            optimiser.zero_grad()
            output = model(X_tensor)
            loss = loss_f(output, y_tensor.reshape(-1, 1))
            total_loss += loss.detach().cpu()
            loss.backward()
            optimiser.step()

        model.eval()
        models.append(copy.deepcopy(model))

        epoch_loss_train.append(getTotalLoss_no_weight(model, loss_f, X_train, y_train, w_train, batch_size))
        epoch_loss_test.append(getTotalLoss_no_weight(model, loss_f, X_test, y_test, w_test, batch_size))
 
        for param_group in optimiser.param_groups:
            lr_epoch= param_group['lr']
            updated_lr = learning_rate_scheduler(epoch=i_epoch, lr_epoch=lr_epoch, initial=lr, epochlist=epoch_loss_train, scheduler=scheduler_type)
            param_group['lr'] = updated_lr
        learning_rate_epochs.append(updated_lr)
        
        
        if epoch_loss_test[-1] < best_loss:
           best_loss = epoch_loss_test[-1]
           best_model = model
           best_epoch = i_epoch
           patience_counter = 0
        else:
           patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    print(">> Training finished")
    print(f"Best model at epoch {best_epoch} with loss of {best_loss}")
    best_model.eval()
    with torch.no_grad():
        output_score = best_model(torch.Tensor(X_test).to(device))
        output_score_train = best_model(torch.Tensor(X_train).to(device))

    return models,epoch_loss_train,epoch_loss_test,output_score,output_score_train, learning_rate_epochs


    
    
    
    



