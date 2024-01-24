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



signal_masses = ["260","270","280","290","300","320","350","400","450","500","550","600","650","700","750","800","900"]
#signal_masses = ["260"]

num_of_features = 12
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

signal_name = "GluGluToRadionToHHTo2G2Tau_M-1000"
auc_for_optimium_features = {}

for signal in signal_masses:
    print(signal)
    signal_df, background_df, combine_df = te.read_dataframes(signal_name = f"GluGluToRadionToHHTo2G2Tau_M-{signal}")

    x_train,x_test = te.getTrainTestSplit(combine_df)
    dict = {}
    epoch = 200

    nodes = [64,64,32]
    models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, feature_list, 0.0001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048, nodes = nodes)

    signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
    roc_auc = auc(fpr, tpr)

    if roc_auc < 0.5:
        roc_auc = 1 - roc_auc

    auc_for_optimium_features[signal] = roc_auc


    plt.figure(figsize=(30,6))
    plt.subplot(1, 3, 1)
    plt.plot(epoch_loss_train, label = 'train')
    plt.plot(epoch_loss_test, label = 'test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(signal_output_score, label = 'Signal')
    plt.hist(background_output_score, label = 'Background')
    plt.xlabel("Output Score")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    with open(f'optimium_features_{num_of_features}.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, auc_for_optimium_features.keys())
        w.writeheader()
        w.writerow(auc_for_optimium_features)
