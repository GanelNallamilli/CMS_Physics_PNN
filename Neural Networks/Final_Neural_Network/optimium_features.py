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

def separate_output_score(output_score,y):
    dict_ = {'pred':output_score.cpu().detach().numpy().flatten(),'true':y}
    temp_df = pd.DataFrame(dict_)
    signal_output_score = temp_df.loc[temp_df['true'] == 1]['pred']
    background_output_score = temp_df.loc[temp_df['true'] == 0]['pred']
    return signal_output_score,background_output_score

num_of_features = 10
feature_list = []
sorted_features = {}

signal_masses = ["260","270","280","290","300","320","350","400","450","500","550","600","650","700","750","800","900"]

for signal_mass in signal_masses:
    signal = "GluGluToRadionToHHTo2G2Tau_M-"+signal_mass
    GluGluToRadionToHHTo2G2Tau_AUC_NN = pd.read_csv(f"{signal}_AUC_NN.csv", index_col = False)
    dict_ = {GluGluToRadionToHHTo2G2Tau_AUC_NN.columns[i]:GluGluToRadionToHHTo2G2Tau_AUC_NN[GluGluToRadionToHHTo2G2Tau_AUC_NN.columns[i]][0] for i in range(len(GluGluToRadionToHHTo2G2Tau_AUC_NN.columns))}
    dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
    list_ = list(dict_.keys())
    sorted_features[signal_mass] = list_



for sig in sorted_features.keys():
    for i in range(num_of_features):
        if sorted_features[sig][i] not in feature_list:
            feature_list.append(sorted_features[sig][i])

print(feature_list)

for i in range(17,19,2):
    num_of_features = i
    feature_list = []
    sorted_features = {}

    for signal_mass in signal_masses:
        signal = "GluGluToRadionToHHTo2G2Tau_M-"+signal_mass
        GluGluToRadionToHHTo2G2Tau_AUC_NN = pd.read_csv(f"{signal}_AUC_NN.csv", index_col = False)
        dict_ = {GluGluToRadionToHHTo2G2Tau_AUC_NN.columns[i]:GluGluToRadionToHHTo2G2Tau_AUC_NN[GluGluToRadionToHHTo2G2Tau_AUC_NN.columns[i]][0] for i in range(len(GluGluToRadionToHHTo2G2Tau_AUC_NN.columns))}
        dict_ = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1], reverse=True)}
        list_ = list(dict_.keys())
        sorted_features[signal_mass] = list_



    for sig in sorted_features.keys():
        for i in range(num_of_features):
            if sorted_features[sig][i] not in feature_list:
                feature_list.append(sorted_features[sig][i])

    print(feature_list)

    auc_for_optimium_features_test = {}
    auc_for_optimium_features_train = {}

    signal_masses = ["320"]

    for signal in signal_masses:
        print(signal)
        signal_df, background_df, combine_df,add_to_test_df = te.read_dataframes(signal_name = f"GluGluToRadionToHHTo2G2Tau_M-{signal}")

        x_train,x_test = te.getTrainTestSplit(combine_df,add_to_test_df)
        dict = {}
        epoch = 500
        lr =0.0001

        nodes = [50,50,50]
        models,epoch_loss_train,epoch_loss_test,output_score,output_score_train, learning_rate_epochs = te.trainNetwork_no_weights(x_train, x_test, feature_list, lr, epoch = epoch, outdir=None, save_models=False, batch_size = 1024, nodes = nodes, model_type='char')

        signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)

        if roc_auc < 0.5:
            roc_auc = 1 - roc_auc

        auc_for_optimium_features_test[signal] = roc_auc
        
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
        fig.suptitle(f'GluGluToRadionToHHTo2G2Tau_M-{signal}-features:{num_of_features}: architecture={nodes}, lr={lr} Testing Data', fontsize=16)
        plt.savefig(f'optimium_features_FIG_{num_of_features}_sig{signal}_lr_{lr}_test.png', format='png')
        plt.show()

        signal_output_score_train,background_output_score_train = separate_output_score(output_score_train,x_train['y'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        fpr, tpr, thresholds = roc_curve(x_train['y'], output_score_train.cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)

        if roc_auc < 0.5:
            roc_auc = 1 - roc_auc

        auc_for_optimium_features_train[signal] = roc_auc
        
        fig, axs = plt.subplots(1, 3, figsize=(24, 10))
        
        axs[0].plot(epoch_loss_train, label = 'train')
        axs[0].plot(epoch_loss_test, label = 'test')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_title('Loss per Epoch')
        axs[0].legend()
        
        axs[1].hist(signal_output_score_train, label = 'Signal',bins=80,histtype='step')
        axs[1].hist(background_output_score_train, label = 'Background',bins=80,histtype='step')
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
        fig.suptitle(f'GluGluToRadionToHHTo2G2Tau_M-{signal}-features:{num_of_features}: architecture={nodes}, lr={lr} Training Data', fontsize=16)
        plt.savefig(f'optimium_features_FIG_{num_of_features}_sig{signal}_lr_{lr}_train.png', format='png')
        plt.show()


    with open(f'optimium_features_v2_{num_of_features}_lr_{lr}_test.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, auc_for_optimium_features_test.keys())
        w.writeheader()
        w.writerow(auc_for_optimium_features_test)

    with open(f'optimium_features_v2_{num_of_features}_lr_{lr}_train.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, auc_for_optimium_features_train.keys())
        w.writeheader()
        w.writerow(auc_for_optimium_features_train)
