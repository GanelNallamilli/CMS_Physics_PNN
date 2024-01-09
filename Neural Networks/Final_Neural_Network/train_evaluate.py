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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def weightedBCELoss(input, target, weight):
  x, y, w = input, target, weight
  log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
  return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))


def read_dataframes(directory = '', signal_name = ''):
    #list of each bkgs for concatenation
    background_list=['DiPhoton', 
                     'TTGG', 
                     'TTGamma',
                     'TTJets',
                     'VBFH_M125',
                     'VH_M125',
                     'WGamma',
                     'ZGamma',
                     'ggH_M125', 
                     'ttH_M125',
                     'GJets']
    
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
                      'ditau_mass']

    df = pd.read_parquet(f'{directory}merged_nominal.parquet')
    with open(f'{directory}summary.json', "r") as f:
        proc_dict = json.load(f)["sample_id_map"]

    signal = df[df.process_id == proc_dict[f"{signal_name}"]]

    listforconc=[]
    for i in background_list:                              
        bkgg = df[df.process_id == proc_dict[i]]
        listforconc.append(bkgg)

    background = pd.concat(listforconc)
    for columns in MinusNineBinning:
            background = background.loc[(background[columns] > -8)]
            signal = signal.loc[(signal[columns] > -8)]


    signal['y']=np.ones(len(signal.index))
    background['y']=np.zeros(len(background.index))

    combine = pd.concat([signal,background])

    return signal,background,combine

def getBatches(arrays, batch_size=None):
    if len(arrays) != 3:
        raise Exception("'getBatchs' arrays should contain 3 variables: x,y,w (data, labels, weights)")
    
    length = len(arrays[0])

    for i in tqdm(range(0, length, batch_size)):
        arrays_batch = [torch.Tensor(array[i:i+batch_size]).to(device) for array in arrays]
        yield arrays_batch

def basicModel(features):
    length = len(features)

    model = nn.Sequential(
        nn.Linear(length, 5),
        nn.ReLU(),
        nn.Linear(5, 1),
        nn.Sigmoid()
        )
    return model

def getTotalLoss(model, loss_f, X, y, w, batch_size):
  total_loss = 0.0

  with torch.no_grad():
    for X_tensor, y_tensor, w_tensor in getBatches([X, y, w], batch_size):
      output = model(X_tensor)
      total_loss += loss_f(output, y_tensor.reshape(-1, 1), w_tensor.reshape(-1, 1)).detach().cpu()



    mean_loss = total_loss / len(X)

  return mean_loss

def getTrainTestSplit(combine_df):
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

    return x_train,x_test

def trainNetwork(train_df, test_df, features, lr,epoch = 200, outdir=None, save_models=False, batch_size = 1024):
    loss_f = lambda x, y, w: weightedBCELoss(x,y,w)

    X_train = train_df[features].to_numpy()
    X_test = test_df[features].to_numpy()
    y_train = train_df["y"].to_numpy()
    y_test = test_df["y"].to_numpy()
    w_train = train_df["weight_central"].to_numpy()
    w_test = test_df["weight_central"].to_numpy()

    model = basicModel(features).to(device)
    optimiser =  torch.optim.Adam(model.parameters(), lr= lr)
    epoch_loss_train = []
    epoch_loss_test = []
    models = []

    print(">> Training...")
    for i_epoch in tqdm(range(0,epoch)):
        print(f"Epoch {i_epoch}")
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

    print(">> Training finished")
    model.eval()
    with torch.no_grad():
        output_score = model(torch.Tensor(X_test).to(device))

    return models,epoch_loss_train,epoch_loss_test,output_score

    
    
    
    



