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
from tqdm.auto import tqdm
import copy
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import random
import os
import joblib
import json
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#loss with weights
def weightedBCELoss(input, target, weight):
  x, y, w = input, target, weight
  log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
  return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))

#loss without weights
def BCELoss(input, target):
  x, y= input, target
  log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
  return torch.mean(-1 * (y*log(x) + (1-y)*log(1-x)))


#Custom Scheduler
def learning_rate_scheduler(epoch, lr_epoch, initial, epochlist, scheduler=None):
    newlr=lr_epoch
    if scheduler == 'Custom':
        if epoch%10 == 0 and epoch > 0:
            f=epochlist[epoch]/epochlist[0]
            newlrr=initial*f
            newlr=float(newlrr.item())
    #if scheduler == 'Linear':
        
    return newlr  

def change_batch(epoch, initial, epochlist):
    f = 1
    if epoch%10 == 0 and epoch > 0:
        f=epochlist[epoch]/epochlist[0]
        f = f + 1
    return f

def read_dataframes(directory = '', signal_name = '',remove_masses = []):
    print(">> Loading data ...")
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


    allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']

    for mass in remove_masses:
        allmasses.remove(str(mass))
        
    sig_pnn_test_conc = []
    listforconc=[]
    for mass in allmasses:                              
        full_sig = df[df.process_id == proc_dict["GluGluToRadionToHHTo2G2Tau_M-"+mass]]
        listforconc.append(full_sig)

    signal = pd.concat(listforconc)

    listforconc=[]
    for i in background_list:                              
        bkgg = df[df.process_id == proc_dict[i]]
        listforconc.append(bkgg)

    background = pd.concat(listforconc)


    signal['y']=np.ones(len(signal.index))
    background['y']=np.zeros(len(background.index))


    listforconc=[]
    add_to_test = ['GJets','TTJets']
    for i in add_to_test:                              
        bkgg = df[df.process_id == proc_dict[i]]
        listforconc.append(bkgg)

    add_to_test_df = pd.concat(listforconc)
    add_to_test_df['y']=np.zeros(len(add_to_test_df.index))

    signal_proportionals = {}
    for mass in allmasses:
        signal_proportionals[mass] = len(signal[signal.MX == int(mass)])/len(signal.MX)


    total_background_length = len(background) + len(add_to_test_df) 
    mass_values_array = []

    for mass in allmasses:
        temp_mass = [int(mass)]*(int(signal_proportionals[mass]*total_background_length)+1)
        for i in range(len(temp_mass)):
            mass_values_array.append(temp_mass[i])

    random.shuffle(mass_values_array)

    background['MX'] = mass_values_array[:len(background)]
    add_to_test_df['MX'] = mass_values_array[len(background):len(background)+len(add_to_test_df)]


    combine = pd.concat([signal,background])

    temp_data_frame = pd.concat([combine,add_to_test_df])
    for col in MinusNineBinning:
        temp_data_frame[col] = temp_data_frame[col].replace(-9, pd.NA)
        column_means = temp_data_frame[col].mean()

        combine[col] = combine[col].replace(-9, pd.NA)
        add_to_test_df[col] = add_to_test_df[col].replace(-9, pd.NA)

        pd.set_option('future.no_silent_downcasting', True)
        
        combine[col] = combine[col].fillna(column_means)
        add_to_test_df[col] = add_to_test_df[col].fillna(column_means)
        
    return signal,background,combine,add_to_test_df

def getWeightedBatches(arrays, batch_size=None):
    weights = arrays[2]
    length = len(weights)
    indices = np.arange(length)


    offset = np.abs(np.min(weights)) if np.min(weights) < 0 else 0
    adjusted_weights = weights + offset
    sampling_prob = adjusted_weights / adjusted_weights.sum()

    for _ in tqdm(range(0, length, batch_size)):
        chosen_indices = np.random.choice(indices, size=batch_size, p=sampling_prob)
        arrays_batch = [torch.Tensor(array[chosen_indices]).to(device) for array in arrays]
        yield arrays_batch


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

def getTotalLoss_no_weight(model, loss_f, X, y, w, batch_size):
  total_loss = 0.0

  with torch.no_grad():
    for X_tensor, y_tensor, w_tensor in getWeightedBatches([X, y, w], batch_size):
      output = model(X_tensor)
      total_loss += loss_f(output, y_tensor.reshape(-1, 1)).detach().cpu()



    mean_loss = total_loss / len(X)

  return mean_loss


def getTrainTestSplit(combine_df,add_to_test_df = [],remove_masses = []):
    print(">> Splitting train/test data ...")
    x_train, x_test, y_train, y_test = train_test_split(combine_df.drop(columns=['y']), combine_df['y'], test_size=(1/3), random_state=42)

    x_train_original = x_train.copy()
    x_test_original = x_test.copy()
    x_train_original['y'] = y_train
    x_test_original['y'] = y_test

    allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']

    for mass in remove_masses:
        allmasses.remove(str(mass))

    int_masses = [int(i) for i in allmasses]
        
    start = 30  # Start value
    end = 1    # End value
    num_elements = 18  # Number of elements in the list
    
    # Calculating step size
    step = (end - start) / (num_elements - 1)
    
    # Generating the list
    list_from_3_to_1 = [start + step * i for i in range(num_elements)]

    
    x_train.loc[(y_train == 1) & (x_train['MX'] == 260), 'weight_central'] *= x_train['weight_central'][y_train == 0].sum() / x_train['weight_central'][(y_train == 1) & (x_train['MX'] == 260)].sum()

    x_test.loc[(y_test == 1) & (x_test['MX'] == 260), 'weight_central'] *= x_test['weight_central'][y_test == 0].sum() / x_test['weight_central'][(y_test == 1) & (x_test['MX'] == 260)].sum()


    for idx, mass in enumerate(int_masses):
        condition = (y_train == 1) & (x_train['MX'] == mass)
        x_train.loc[condition, 'weight_central'] *= x_train['weight_central'][(y_train == 1) & (x_train['MX'] == 260)].sum() / x_train['weight_central'][condition].sum()


    for idx, mass in enumerate(int_masses):
        condition = (y_test == 1) & (x_test['MX'] == mass)
        x_test.loc[condition, 'weight_central'] *= x_test['weight_central'][(y_test == 1) & (x_test['MX'] == 260)].sum() / x_test['weight_central'][condition].sum()


    for idx, mass in enumerate(int_masses):
        condition = (y_train == 1) & (x_train['MX'] == mass)
        x_train.loc[condition, 'weight_central'] *= list_from_3_to_1[idx]


    for idx, mass in enumerate(int_masses):
        condition = (y_test == 1) & (x_test['MX'] == mass)
        x_test.loc[condition, 'weight_central'] *= list_from_3_to_1[idx]

        

    x_train.loc[y_train == 1, 'weight_central'] *= x_train['weight_central'][y_train == 0].sum() / x_train['weight_central'][y_train == 1].sum()
    x_train['weight_central'] *= len(x_train['weight_central']) / x_train['weight_central'].sum()


    x_test.loc[y_test == 1, 'weight_central'] *= x_test['weight_central'][y_test == 0].sum() / x_test['weight_central'][y_test == 1].sum()
    x_test['weight_central'] *= len(x_test['weight_central']) / x_test['weight_central'].sum()


    y_train = pd.DataFrame(y_train,columns=['y'])
    y_test = pd.DataFrame(y_test,columns=['y'])

    x_train['y'] = y_train
    x_test['y'] = y_test

    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    x_test = pd.concat([x_test, add_to_test_df], ignore_index=True)

    x_train = x_train.sample(frac=1).reset_index(drop=True)
    x_test = x_test.sample(frac=1).reset_index(drop=True)

    return x_train,x_test,x_train_original,x_test_original


def trainNetwork_no_weights(train_df, test_df, features, lr,epoch = 200, save_models=False, batch_size = 1024,nodes = [5],patience = 5 ,model_type = 'basic',scheduler_type='None',directory = '',scheduler = True, dynamic_batch = False, single_mass = False):
    loss_f = lambda x, y: BCELoss(x,y)

    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(">> Directory created successfully ...")
    else:
        directory_update = directory + '_run_2'
        i = 3
        while os.path.exists(directory_update):
            directory_update = directory + f'_run_{i}'
            i += 1
        directory = directory_update
        os.makedirs(directory)
        print(f">> Directory exists, created directory: {directory} ...")

    with open(f'{directory}/test_df.pkl', 'wb') as f:
            pickle.dump(test_df, f)

    with open(f'{directory}/train_df.pkl', 'wb') as f:
            pickle.dump(train_df, f)

    model_info = {
        'starting_learning_rate': lr,
        'maximum_epoch': epoch,
        'batch_size': batch_size,
        'Architecture': nodes,
        'patience': patience,
        'model_type': model_type,
        'scheduler_type': scheduler_type,
        'scheduler_status': scheduler
    }

    with open(f'{directory}/model_info.json', 'w') as json_file:
        json.dump(model_info, json_file, indent=4)

    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    joblib.dump(scaler, f'{directory}/scaler.save')

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
    
    optimiser =  torch.optim.Adam(model.parameters(), lr= lr)
    epoch_loss_train = []
    epoch_loss_test = []
    models = []
    best_model = ""


    patience = patience
    best_loss = float('inf')
    patience_counter = 0
    learning_rate_epochs=[]
    batch_size_epochs=[]
    allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']
    signal_output_score_arr_test = [[] for i in range(len(allmasses))]
    signal_output_score_arr_train = [[] for i in range(len(allmasses))]
    signal_output_score_test_dict = {}
    signal_output_score_train_dict = {}
    best_epoch = 0
    
    print(">> Training...")
    for i_epoch in tqdm(range(0,epoch)):
        print(f"Epoch {i_epoch}")
        total_loss = 0.0
        model.train()
        # if i_epoch%250 == 0:
        #     print(f'Epoch: {i_epoch}' )
        batch_gen = getWeightedBatches([X_train, y_train, w_train], batch_size = batch_size)

        for X_tensor, y_tensor, w_tensor in batch_gen:
            optimiser.zero_grad()
            output = model(X_tensor)
            loss = loss_f(output, y_tensor.reshape(-1, 1))
            total_loss += loss.detach().cpu()
            loss.backward()
            optimiser.step()

        if save_models:
            torch.save(model, f'{directory}/model_epoch_{i_epoch}.pth')

        model.eval()
        models.append(copy.deepcopy(model))

        print(">> Getting train/loss for entire pNN ...")
        epoch_loss_train.append(getTotalLoss_no_weight(model, loss_f, X_train, y_train, w_train, batch_size))
        epoch_loss_test.append(getTotalLoss_no_weight(model, loss_f, X_test, y_test, w_test, batch_size))

        

        np.save(f'{directory}/epoch_loss_train.npy', epoch_loss_train)
        np.save(f'{directory}/epoch_loss_test.npy', epoch_loss_test)

        print(f'Train Epoch Loss: {epoch_loss_train[-1]}')
        print(f'Test Epoch Loss: {epoch_loss_test[-1]}')
        
        if scheduler:
            for param_group in optimiser.param_groups:
                lr_epoch= param_group['lr']
                updated_lr = learning_rate_scheduler(epoch=i_epoch, lr_epoch=lr_epoch, initial=lr, epochlist=epoch_loss_train, scheduler=scheduler_type)
                param_group['lr'] = updated_lr
                print(f"Learning rate: {param_group['lr']}")
            learning_rate_epochs.append(updated_lr)

        factor = change_batch(epoch=i_epoch, initial=lr, epochlist=epoch_loss_train)

        if dynamic_batch:
            if batch_size*int(factor) < 131072:
                batch_size = int(batch_size*factor)
                print(f'Batch size: {batch_size}')
    
            batch_size_epochs.append(batch_size)

        

        np.save(f'{directory}/learning_rate_epochs.npy', learning_rate_epochs)
        np.save(f'{directory}/batch_size_epochs.npy', batch_size_epochs)
        
        
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

        if best_model == "":
            best_model = model

        np.save(f'{directory}/best_epoch.npy', best_epoch)

        if single_mass:

            print(">> Getting train/loss for individual signals using pNN ...")
            for i,mass_eval in enumerate(allmasses):
    
                signal_mass = int(mass_eval)
    
                signal_df = test_df[(test_df.y == 1) & (test_df.MX == signal_mass)]
                background_df = test_df[(test_df.y == 0)]
                background_df.loc[:, 'MX'] = signal_mass
                specific_sig_combine_test_data = pd.concat([signal_df,background_df])
                specific_sig_test_df = specific_sig_combine_test_data.sample(frac=1,random_state = 42).reset_index(drop=True)
                
                signal_output_score_arr_test[i].append(getTotalLoss_no_weight(model, loss_f, specific_sig_test_df[features].to_numpy(), specific_sig_test_df['y'].to_numpy(), specific_sig_test_df['weight_central'].to_numpy(), batch_size))
    
                signal_output_score_test_dict[signal_mass] = signal_output_score_arr_test[i]
    
                
    
                signal_df = train_df[(train_df.y == 1) & (train_df.MX == signal_mass)]
                background_df = train_df[(train_df.y == 0)]
                background_df.loc[:, 'MX'] = signal_mass
                specific_sig_combine_train_data = pd.concat([signal_df,background_df])
                specific_sig_train_df = specific_sig_combine_train_data.sample(frac=1,random_state = 42).reset_index(drop=True)
                
                signal_output_score_arr_train[i].append(getTotalLoss_no_weight(model, loss_f, specific_sig_train_df[features].to_numpy(), specific_sig_train_df['y'].to_numpy(), specific_sig_train_df['weight_central'].to_numpy(), batch_size))
    
                signal_output_score_train_dict[signal_mass] = signal_output_score_arr_train[i]
            
        with open(f'{directory}/signal_output_score_train_dict.pkl', 'wb') as f:
            pickle.dump(signal_output_score_train_dict, f)

        with open(f'{directory}/signal_output_score_test_dict.pkl', 'wb') as f:
            pickle.dump(signal_output_score_test_dict, f)


    print(">> Training finished")
    print(f"Best model at epoch {best_epoch} with loss of {best_loss}")
    best_model.eval()
    with torch.no_grad():
        output_score = best_model(torch.Tensor(X_test).to(device))
        output_score_train = best_model(torch.Tensor(X_train).to(device))

    return models,epoch_loss_train,epoch_loss_test,output_score,output_score_train, learning_rate_epochs,scaler,best_epoch,directory,signal_output_score_train_dict,signal_output_score_test_dict




