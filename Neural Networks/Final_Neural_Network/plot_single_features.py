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



feature_info = pd.read_csv("ROC_feature_info.csv")

signal_df, background_df, combine_df = te.read_dataframes(signal_name = 'GluGluToRadionToHHTo2G2Tau_M-1000')

x_train,x_test = te.getTrainTestSplit(combine_df)
feature_list = ['Diphoton_mass', 'Diphoton_pt_mgg', 'Diphoton_dPhi',
       'LeadPhoton_pt_mgg', 'SubleadPhoton_pt_mgg', 'MET_pt',
       'diphoton_met_dPhi', 'ditau_met_dPhi', 'ditau_deta', 'lead_lepton_pt',
       'lead_lepton_mass', 'jet_1_pt', 'ditau_pt', 'ditau_mass',
       'ditau_dR', 'ditau_dphi', 'Diphoton_ditau_dphi', 'dilep_leadpho_mass','reco_MX_mgg',
       'Diphoton_ditau_deta', 'Diphoton_lead_lepton_deta',
       'Diphoton_lead_lepton_dR', 'Diphoton_sublead_lepton_deta',
       'Diphoton_sublead_lepton_dR', 'LeadPhoton_ditau_dR',
       'LeadPhoton_lead_lepton_dR', 'SubleadPhoton_lead_lepton_dR']

dict = {}
for name in feature_list:
    epoch = 200
    models,epoch_loss_train,epoch_loss_test,output_score = te.trainNetwork(x_train, x_test, [name], 0.001, epoch = epoch, outdir=None, save_models=False, batch_size = 2048)

    test= np.linspace(min(combine_df[name]),max(combine_df[name]),10000).reshape(-1,1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
            output_score_test = models[-1](torch.Tensor(test).to(device))

    signal_output_score,background_output_score = separate_output_score(output_score,x_test['y'])

    fpr, tpr, thresholds = roc_curve(x_test['y'], output_score.cpu().detach().numpy())
    roc_auc = auc(fpr, tpr)

    dict[name] = roc_auc


    feature_info_row = feature_info.loc[feature_info['Feature'] == name]

    fig, ax1 = plt.subplots()
    color = 'black'
    ax1.set_xlabel(name)
    ax1.set_ylabel('Frequency', color=color)
    ax1.hist(background_df[name], bins=80, range=(feature_info_row['Range_Low'].item(), feature_info_row['Range_High'].item()), alpha = 0.8,label = 'Background')
    ax1.hist(signal_df[name], bins=80, range=(feature_info_row['Range_Low'].item(), feature_info_row['Range_High'].item()), alpha = 0.8,label = 'Signal')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Output Score', color=color)  # we already handled the x-label with ax1
    ax2.plot(test, output_score_test.cpu().detach().numpy().flatten(), color=color)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

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


with open('GluGluToRadionToHHTo2G2Tau_M_1000_AUC_NN.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, dict.keys())
    w.writeheader()
    w.writerow(dict)





