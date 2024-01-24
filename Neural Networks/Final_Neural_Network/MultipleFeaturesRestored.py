# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:42:20 2024

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
# =============================================================================
# """
# ignore automation
# """
# signal_names=["GluGluToRadionToHHTo2G2Tau_M-260",
#             "GluGluToRadionToHHTo2G2Tau_M-270",
#             "GluGluToRadionToHHTo2G2Tau_M-280",
#             "GluGluToRadionToHHTo2G2Tau_M-290",
#             "GluGluToRadionToHHTo2G2Tau_M-300",
#             "GluGluToRadionToHHTo2G2Tau_M-320",
#             "GluGluToRadionToHHTo2G2Tau_M-350",
#             "GluGluToRadionToHHTo2G2Tau_M-400",
#             "GluGluToRadionToHHTo2G2Tau_M-450",
#             "GluGluToRadionToHHTo2G2Tau_M-500",
#             "GluGluToRadionToHHTo2G2Tau_M-550",
#             "GluGluToRadionToHHTo2G2Tau_M-600",
#             "GluGluToRadionToHHTo2G2Tau_M-650",
#             "GluGluToRadionToHHTo2G2Tau_M-700",
#             "GluGluToRadionToHHTo2G2Tau_M-750",
#             "GluGluToRadionToHHTo2G2Tau_M-800",
#             "GluGluToRadionToHHTo2G2Tau_M-900",
#             "GluGluToRadionToHHTo2G2Tau_M-1000"]
# 
# signal_names=["GluGluToRadionToHHTo2G2Tau_M-1000",
#             "GluGluToRadionToHHTo2G2Tau_M-600",
#             "GluGluToRadionToHHTo2G2Tau_M-400"]
# 
# 
# m1000=['reco_MX_mgg','Diphoton_pt_mgg','ditau_pt','LeadPhoton_pt_mgg','dilep_leadpho_mass','ditau_dR','Diphoton_dPhi','ditau_dphi','SubleadPhoton_pt_mgg','ditau_met_dPhi']
# m600=['reco_MX_mgg','Diphoton_pt_mgg','ditau_pt','LeadPhoton_pt_mgg','ditau_dR','Diphoton_dPhi','ditau_dphi','dilep_leadpho_mass','ditau_met_dPhi','Diphoton_ditau_dphi']
# m400=['reco_MX_mgg','ditau_mass','ditau_dR','ditau_met_dPhi','Diphoton_ditau_dphi','Diphoton_pt_mgg','jet_1_pt','Diphoton_dPhi','ditau_pt']
# 
# #feature_list=['Diphoton_pt_mgg','LeadPhoton_pt_mgg','reco_MX_mgg','ditau_pt','dilep_leadpho_mass','MET_pt',
# #                'ditau_dR','SubleadPhoton_pt_mgg','SubleadPhoton_pt_mgg','lead_lepton_pt'] 
# 
# feature_list=[m1000,m600,m400]  
# l=0
# auclist=[]
# for ben in signal_names:
#     
#     feature_info = pd.read_csv("ROC_feature_info.csv")
#     
#     signal_df, background_df, combine_df = te.read_dataframes(signal_name = ben)
#     
#     x_train,x_test = te.getTrainTestSplit(combine_df)
#     feature_list = ['Diphoton_mass', 'Diphoton_pt_mgg', 'Diphoton_dPhi',
#            'LeadPhoton_pt_mgg', 'SubleadPhoton_pt_mgg', 'MET_pt',
#            'diphoton_met_dPhi', 'ditau_met_dPhi', 'ditau_deta', 'lead_lepton_pt',
#            'lead_lepton_mass', 'jet_1_pt', 'ditau_pt', 'ditau_mass',
#            'ditau_dR', 'ditau_dphi', 'Diphoton_ditau_dphi', 'dilep_leadpho_mass','reco_MX_mgg',
#            'Diphoton_ditau_deta', 'Diphoton_lead_lepton_deta',
#            'Diphoton_lead_lepton_dR', 'Diphoton_sublead_lepton_deta',
#            'Diphoton_sublead_lepton_dR', 'LeadPhoton_ditau_dR',
#            'LeadPhoton_lead_lepton_dR', 'SubleadPhoton_lead_lepton_dR']
#     
#   
#     
#     epoch = 200
#     models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, 0.001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048)
#     
# 
#     
#     signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])
#     
#     fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
#     roc_auc = auc(fpr, tpr)
#     print(f'Printing AUC Score:{roc_auc} for signal:{ben}')
#     
#     fig, axs = plt.subplots(1, 3, figsize=(24, 10))
#    # plt.figure(figsize=(30,6))
#    # plt.subplot(1, 3, 1)
#     axs[0].plot(epoch_loss_train, label = 'train')
#     axs[0].plot(epoch_loss_test, label = 'test')
#     axs[0].set_ylabel('Loss')
#     axs[0].set_xlabel('Epoch')
#     axs[0].legend()
#     
#    # plt.subplot(1, 3, 2)
#     axs[1].hist(signal_output_score, label = 'Signal',bins=80,histtype='step')
#     axs[1].hist(background_output_score, label = 'Background',bins=80,histtype='step')
#     axs[1].set_xlabel("Output Score")
#     axs[1].set_ylabel("Frequency")
#     axs[1].legend()
#     
#   #  plt.subplot(1, 3, 3)
#     axs[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
#     axs[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     axs[2].set_xlim([0.0, 1.0])
#     axs[2].set_ylim([0.0, 1.05])
#     axs[2].set_xlabel('False Positive Rate')
#     axs[2].set_ylabel('True Positive Rate')
#     axs[2].set_title('Receiver Operating Characteristic',fontsize=18)
#     axs[2].legend(loc="lower right")
#     plt.tight_layout()
#     #plt.title(f'{signal_names[l]}')
#     fig.suptitle(f'{signal_names[l]}', fontsize=16)
#     plt.show()
#     
#     auclist.append([signal_names[l],roc_auc])
#     l+=1
# =============================================================================
    # =============================================================================
    # with open('GluGluToRadionToHHTo2G2Tau_M_800_AUC_NN.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    #     w = csv.DictWriter(f, dict.keys())
    #     w.writeheader()
    #     w.writerow(dict)
    # =============================================================================
#%%



allmasses=['260','270','280','290','300','320','350','400','450','500','550','600','650','700','750','800','900','1000']
#top_number=3 # select number of features
top_numbers=[3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,27]


all_features=pd.read_csv('ORDEREDFEATURESUITABILITY.csv') #importing csv file that contains all features 
                                                                #in their order of highest AUC score for their corresponding masses
for mass in allmasses:
    signal_names="GluGluToRadionToHHTo2G2Tau_M-"+mass #choose signal to analyse
    auclist=[]
    for top_number in top_numbers:         
        feature_info = pd.read_csv("ROC_feature_info.csv")
        
        signal_df, background_df, combine_df = te.read_dataframes(signal_name = signal_names)
        
        x_train,x_test = te.getTrainTestSplit(combine_df)
        
        feature_list = all_features[signal_names].tolist()[0:top_number]
        
        
        epoch = 200
        models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, 0.001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048)
        
        
        
        signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])
        
        fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)
        print(f'Printing AUC Score:{roc_auc} for signal:{signal_names}')
        
        fig, axs = plt.subplots(1, 3, figsize=(24, 10))
           
        axs[0].plot(epoch_loss_train, label = 'train')
        axs[0].plot(epoch_loss_test, label = 'test')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].legend()
        
        axs[1].hist(signal_output_score, label = 'Signal',bins=80,histtype='step')
        axs[1].hist(background_output_score, label = 'Background',bins=80,histtype='step')
        axs[1].set_xlabel("Output Score")
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
        fig.suptitle(f'{signal_names}', fontsize=16)
        plt.show()
        
        auclist.append([signal_names,roc_auc,f'{len(feature_list)}'])
        
    featurescore_df = pd.DataFrame(auclist, columns=['signal', 'score', 'number_of_features'])


    output_file_path = f'Neural Networks\\Final_Neural_Network\\csv of numberfeatures against auc score\\NumFeatures_AUCSCORE_{signal_names}.csv'
    featurescore_df.to_csv(output_file_path, index=False)

#%%

featurescore_df = pd.DataFrame(auclist, columns=['signal', 'score', 'number_of_features'])


output_file_path = f'Neural Networks\\Final_Neural_Network\\csv of numberfeatures against auc score\\NumFeatures_AUCSCORE_{signal_names}.csv'
featurescore_df.to_csv(output_file_path, index=False)

#%%
mass='1000'
signal_names="GluGluToRadionToHHTo2G2Tau_M-"+mass
featurescore= np.loadtxt(f'Neural Networks\\Final_Neural_Network\\csv of numberfeatures against auc score\\NumFeatures_AUCSCORE_{signal_names}.csv',
                 delimiter=",",skiprows=1,dtype=str)
minlim=float(min(featurescore[:,1]))-0.1
plt.scatter(featurescore[:,2].astype(float),featurescore[:,1].astype(float))
plt.xlabel('Number of training features')
plt.ylabel('AUC Score')
plt.ylim(minlim,1)
plt.grid()
plt.title(f'AUC Scores: training range of features: {signal_names}',fontsize=18)
plt.savefig(f'Neural Networks\\Final_Neural_Network\\csv of numberfeatures against auc score\\NumFeatures_AUCSCORE_{signal_names}.png')
