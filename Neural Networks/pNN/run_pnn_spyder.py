# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 00:14:38 2024

@author: drpla
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
import train_evaluate_pnn as te
import csv 
import os
import matplotlib.gridspec as gridspec

hep.style.use("CMS")
auclist=[]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def separate_output_score(output_score,y):
    dict_ = {'pred':output_score.cpu().detach().numpy().flatten(),'true':y}
    temp_df = pd.DataFrame(dict_)
    signal_output_score = temp_df.loc[temp_df['true'] == 1]['pred']
    background_output_score = temp_df.loc[temp_df['true'] == 0]['pred']
    return signal_output_score,background_output_score

#%%


#SET NUMBER OF FEATURES HERE
num_of_features = 18



feature_list = []
sorted_features = {}

signal_masses = ["260","270","280","290","300","320","350","400","450","500","550","600","650","700","750","800","900"]

for signal_mass in signal_masses:
    signal = "GluGluToRadionToHHTo2G2Tau_M-"+signal_mass
    GluGluToRadionToHHTo2G2Tau_AUC_NN = pd.read_csv(f"Feature_list/{signal}_AUC_NN.csv", index_col = False, on_bad_lines='skip')
    dict_ = {GluGluToRadionToHHTo2G2Tau_AUC_NN.columns[i]:GluGluToRadionToHHTo2G2Tau_AUC_NN[GluGluToRadionToHHTo2G2Tau_AUC_NN.columns[i]][0] for i in range(len(GluGluToRadionToHHTo2G2Tau_AUC_NN.columns))}
    dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
    list_ = list(dict_.keys())
    sorted_features[signal_mass] = list_


for sig in sorted_features.keys():
    for i in range(num_of_features):
        if sorted_features[sig][i] not in feature_list:
            feature_list.append(sorted_features[sig][i])

feature_list = feature_list + ['MX']
print(feature_list)

#%%


architectures= [[50,50,50]] 

learningrate = 0.0001

plot_learning_rate='yes'
scheduler_type='Custom'

x_test_global = None


auclist=[]
for nodes in architectures:
    signal_df, background_df, combine_df, add_to_test_df = te.read_dataframes()

    x_train,x_test = te.getTrainTestSplit(combine_df,add_to_test_df)

    x_test_global = x_test.copy()
    #x_train,x_test = te.getTrainTestSplit(combine_df)
    
    epoch = 500
#        models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, 0.001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048)

    models,epoch_loss_train,epoch_loss_test,output_score,output_score_train, learning_rate_epochs = te.trainNetwork_no_weights(x_train, x_test, feature_list, learningrate, epoch = epoch, outdir=None, save_models=False,
                        batch_size = 2048*4, nodes = nodes, model_type='char',scheduler_type=scheduler_type)

    # Specify the directory path
    directory = f'models_{num_of_features}_features'
    
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")

    for i in range(len(models)):
        torch.save(models[i], f'models_{num_of_features}_features/model_epoch_{i}.pth')     
    
    
    
        signal_output_score_train,background_output_score_train = separate_output_score(output_score_train,x_train['y'])
        signal_output_score_test,background_output_score_test = separate_output_score(output_score,x_test['y'])

        
        #training aucscore
        fpr_train, tpr_train, thresholds_train = roc_curve(x_train['y'], output_score_train.cpu().detach().numpy())
        roc_auc_train = auc(fpr_train, tpr_train)
    #    print(f'Printing TRAINING AUC Score:{roc_auc_train} for signal {signal_names} of architecture {(nodes)}')
        
        #testing aucscore
        fpr_test, tpr_test, thresholds_test = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
        roc_auc_test = auc(fpr_test, tpr_test)
     #   print(f'Printing TESTING AUC Score:{roc_auc_test} for signal {signal_names} of architecture {(nodes)}')
        
        
        fig = plt.figure(figsize=(24, 10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

        ax0 = plt.subplot(gs[:, 0])
        ax2 = plt.subplot(gs[:, 2])
        ax3 = plt.subplot(gs[1, 1])
        ax1 = plt.subplot(gs[0, 1])



           
        line_train,=ax0.plot(epoch_loss_train, color='darkorange', label = 'train')
        line_test,=ax0.plot(epoch_loss_test, color='darkblue', label = 'test')
       # ax0.plot(label='learning rate',color='red')
        ax0.set_ylabel('Loss')
        ax0.set_xlabel('Epoch')
        ax0.set_title('Loss per Epoch')
       # ax0.legend(loc='upper right',fontsize=16)
        lines = [line_train,line_test]
        if plot_learning_rate == 'yes':
            ax1_twin = ax0.twinx()
            line_lr,=ax1_twin.plot(learning_rate_epochs, label='learning rate', linestyle='--', color='red')
            ax1_twin.set_ylabel('Learning Rate')
            ax1_twin.tick_params(axis='y')
            lines = [line_train,line_test,line_lr]
        
        labels = [line.get_label() for line in lines]
        ax1_twin.legend(lines, labels, loc='upper right', fontsize=16)    

            

        ax1.hist(signal_output_score_train, label = 'Train Signal',bins=80,histtype='step')
        ax1.hist(background_output_score_train, label = 'Test Background',bins=80,histtype='step')
       # ax1.set_xlabel("Output Score")
        ax1.set_title('Classification Distribution')
        ax1.set_ylabel("Frequency")
        ax1.legend(fontsize=16)
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_xticklabels([])
        
        

        
        ax3.hist(signal_output_score_test, label = 'Test Signal',bins=80,histtype='step')
        ax3.hist(background_output_score_test, label = 'Test Background',bins=80,histtype='step')
        ax3.set_xlabel("Output Score")
        #ax3.set_title('Classification Distribution')
        #ax3.set_ylabel("Frequency")
        ax3.set_xlim([-0.05, 1.05])
        
        ax3.legend(fontsize=16)
        
        ax2.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'train ROC curve (area = {roc_auc_train:.3f})')
        ax2.plot(fpr_test, tpr_test, color='darkblue', lw=2, label=f'test ROC curve (area = {roc_auc_test:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic',fontsize=18)
        ax2.legend(loc="lower right",fontsize=16)
        
        plt.tight_layout()
        #plt.title(f'{signal_names[l]}')
        fig.suptitle(f'{signal_names}: architecture={nodes}, lr={learningrate}', fontsize=16)
        gs.update(hspace=0)
  #      plt.savefig(f'FinalNon_p_NN_PLOTS_DATA/a11022024plots_{mass}_arch={nodes}_init_lr={learningrate}.png')

        plt.show()    
featurescore_df = pd.DataFrame(auclist, columns=['score', 'nodes per hidden layer'])
    
    #output_file_path = f'OptimisedArch310124_{mass}_lr{learningrate}_[50,50,50].csv'
    #featurescore_df.to_csv(output_file_path, index=False)
         
