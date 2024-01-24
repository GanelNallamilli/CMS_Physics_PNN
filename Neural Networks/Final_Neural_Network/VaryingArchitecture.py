# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:12:11 2024

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
import train_evaluate as te
import csv 
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


architectures=[[8,16,8],
               [16,16,8],
               [16,32,16],
               [32,32,16],
               [32,64,32],
               [64,64,32],
               [64,128,64],
               [128,128,64],
               [16,32,16,8],
               [32,32,16,8],
               [32,64,32,8],
               [64,64,32,8],
               [64,128,64,8],
               [128,128,64,8]]



learningrate = 0.01               
top_number=27
               
all_features=pd.read_csv('ORDEREDFEATURESUITABILITY.csv')

allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']

for mass in allmasses:
    signal_names="GluGluToRadionToHHTo2G2Tau_M-"+mass #choose signal to analyse
    auclist=[]
    for nodes in architectures:
        feature_info = pd.read_csv("ROC_feature_info.csv")
        
        signal_df, background_df, combine_df = te.read_dataframes(signal_name = signal_names)
        
        x_train,x_test = te.getTrainTestSplit(combine_df)
        
        feature_list = all_features[signal_names].tolist()[0:top_number]
        
        
        epoch = 200
    #        models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, 0.001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048)
    
        models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, learningrate, epoch = epoch, outdir=None, save_models=False, batch_size = 2048, nodes = nodes)        
        
        
        signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])
        
        fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)
        print(f'Printing AUC Score:{roc_auc} for signal {signal_names} of architecture {(nodes)}')
        
        fig, axs = plt.subplots(1, 3, figsize=(24, 10))
           
        axs[0].plot(epoch_loss_train, label = 'train')
        axs[0].plot(epoch_loss_test, label = 'test')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_title('Loss per Epoch')
        axs[0].legend()
        
        axs[1].hist(signal_output_score, label = 'Signal',bins=80,histtype='step')
        axs[1].hist(background_output_score, label = 'Background',bins=80,histtype='step')
        axs[1].set_xlabel("Output Score")
        axs[1].set_title('Classification Distribution')
        axs[1].set_ylabel("Frequency")
        axs[1].legend()
        
        axs[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        axs[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[2].set_xlim([0.0, 1.0])
        axs[2].set_ylim([0.0, 1.05])
        axs[2].set_xlabel('False Positive Rate')
        axs[2].set_ylabel('True Positive Rate')
        axs[2].set_title('Receiver Operating Characteristic',fontsize=18)
        axs[2].legend(loc="lower right")
        plt.tight_layout()
        #plt.title(f'{signal_names[l]}')
        fig.suptitle(f'{signal_names}: architecture={nodes}, lr={learningrate}', fontsize=16)
        plt.show()
        plt.savefig(f'Neural Networks\\Final_Neural_Network\\VaryingArchitectures\\ExamplePlots\\lossclassROC_m={mass}_arch={nodes}_lr={learningrate}.png')
    
        
        auclist.append([signal_names,roc_auc,nodes])
        
    featurescore_df = pd.DataFrame(auclist, columns=['signal', 'score', 'nodes per hidden layer'])
    
    output_file_path = f'Neural Networks\\Final_Neural_Network\\VaryingArchitectures\\Architectures_{mass}_lr{learningrate}.csv'
    featurescore_df.to_csv(output_file_path, index=False)
         
#%%


mass='260'
signal_names="GluGluToRadionToHHTo2G2Tau_M-"+mass
architectures=[[8,16,8],
               [16,16,8],
               [16,32,16],
               [32,32,16],
               [32,64,32],
               [64,64,32],
               [64,128,64],
               [128,128,64],
               [16,32,16,8],
               [32,32,16,8],
               [32,64,32,8],
               [64,64,32,8],
               [64,128,64,8],
               [128,128,64,8]]

learningrate = 0.01               
top_number=27
               
all_features=pd.read_csv('ORDEREDFEATURESUITABILITY.csv')



for nodes in architectures:
    feature_info = pd.read_csv("ROC_feature_info.csv")
    
    signal_df, background_df, combine_df = te.read_dataframes(signal_name = signal_names)
    
    x_train,x_test = te.getTrainTestSplit(combine_df)
    
    feature_list = all_features[signal_names].tolist()[0:top_number]
    
    
    epoch = 200
#        models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, 0.001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048)

    models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, learningrate, epoch = epoch, outdir=None, save_models=False, batch_size = 2048, nodes = nodes)        
    
    
    signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])
    
    fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
    roc_auc = auc(fpr, tpr)
    print(f'Printing AUC Score:{roc_auc} for signal {signal_names} of architecture {(nodes)}')
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
       
    axs[0].plot(epoch_loss_train, label = 'train')
    axs[0].plot(epoch_loss_test, label = 'test')
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_title('Loss per Epoch')
    axs[0].legend()
    
    axs[1].hist(signal_output_score, label = 'Signal',bins=80,histtype='step')
    axs[1].hist(background_output_score, label = 'Background',bins=80,histtype='step')
    axs[1].set_xlabel("Output Score")
    axs[1].set_title('Classification Distribution')
    axs[1].set_ylabel("Frequency")
    axs[1].legend()
    
    axs[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    axs[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axs[2].set_xlim([0.0, 1.0])
    axs[2].set_ylim([0.0, 1.05])
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_ylabel('True Positive Rate')
    axs[2].set_title('Receiver Operating Characteristic',fontsize=18)
    axs[2].legend(loc="lower right")
    plt.tight_layout()
    #plt.title(f'{signal_names[l]}')
    fig.suptitle(f'{signal_names}: architecture={nodes}, lr={learningrate}', fontsize=16)
    plt.show()
    plt.savefig(f'Neural Networks\\Final_Neural_Network\\VaryingArchitectures\\ExamplePlots\\lossclassROC_m={mass}_arch={nodes}_lr={learningrate}.png')

    
    auclist.append([signal_names,roc_auc,nodes])
    
featurescore_df = pd.DataFrame(auclist, columns=['signal', 'score', 'nodes per hidden layer'])

output_file_path = f'Neural Networks\\Final_Neural_Network\\VaryingArchitectures\\Architectures_{mass}_lr{learningrate}.csv'
featurescore_df.to_csv(output_file_path, index=False)



#%%

filenames=['Architectures_260_lr0.01.csv',
 'Architectures_270_lr0.01.csv',
 'Architectures_280_lr0.01.csv',
 'Architectures_290_lr0.01.csv',
 'Architectures_300_lr0.01.csv',
 'Architectures_320_lr0.01.csv',
 'Architectures_350_lr0.01.csv',
 'Architectures_400_lr0.01.csv',
 'Architectures_450_lr0.01.csv',
 'Architectures_500_lr0.01.csv',
 'Architectures_550_lr0.01.csv',
 'Architectures_600_lr0.01.csv',
 'Architectures_650_lr0.01.csv',
 'Architectures_700_lr0.01.csv',
 'Architectures_750_lr0.01.csv',
 'Architectures_800_lr0.01.csv',
 'Architectures_900_lr0.01.csv',
 'Architectures_1000_lr0.01.csv']
#%%

architecture_nodes='[128, 128, 64]'
columns = ['signal','score','nodes per hidden layer']
df_featurescore=pd.DataFrame(columns=columns)

for file in filenames:
    df = pd.read_csv(f'Neural Networks\\Final_Neural_Network\\VaryingArchitectures\\{file}')
    #print(df)
    df_featurescore = pd.concat([df_featurescore,df], ignore_index=True)

filtered_df = df_featurescore[df_featurescore['nodes per hidden layer'].apply(lambda x: x == architecture_nodes)]


plt.scatter(allmasses,filtered_df['score'],label=architecture_nodes)
plt.xlabel('Mass',fontsize=18)
plt.ylabel('AUC Score',fontsize=18)
plt.ylim(0,1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
plt.legend(loc='lower right',fontsize=10)
plt.title(f'AUC score against mass: Architecture = {architecture_nodes}',fontsize=18)
