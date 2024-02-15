#Load packages
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

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

signal_name_ = 'GluGluToRadionToHHTo2G2Tau_M-400'
signal_df, background_df, combine_df = read_dataframes(signal_name = signal_name_)


import numpy as np
import matplotlib.pyplot as plt

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
  print(f'{name} graph:')

  signal_data = np.array(signal_df[name])
  background_data = np.array(background_df[name])
  signal_weights = np.array(signal_df['weight_central'])
  background_weights = np.array(background_df['weight_central'])

  # Combine and sort data
  data = np.concatenate((signal_data, background_data))
  weights = np.concatenate((signal_weights, background_weights))
  labels = np.concatenate((np.ones(len(signal_data)), np.zeros(len(background_data))))

  # Sort by data in descending order
  sorted_indices = np.argsort(-data)
  sorted_labels = labels[sorted_indices]
  sorted_weights = weights[sorted_indices]

  # Compute cumulative sums of weights for positives and negatives
  cumulative_positive_weights = np.cumsum(sorted_weights * sorted_labels)
  cumulative_negative_weights = np.cumsum(sorted_weights * (1 - sorted_labels))

  # Unique thresholds (data values), maintaining descending order
  unique_thresholds, unique_indices = np.unique(data, return_inverse=True)

  # Compute TPR and FPR at unique thresholds using bincount
  total_positive_weight = cumulative_positive_weights[-1]
  total_negative_weight = cumulative_negative_weights[-1]
  tpr_weights = (sorted_weights * sorted_labels)/data
  fpr_weights = (sorted_weights * (1 - sorted_labels))/data
  #tpr = np.cumsum(np.bincount(data, weights=tpr_weights.astype(float))) / total_positive_weight
  #fpr = np.cumsum(np.bincount(data, weights=fpr_weights.astype(float))) / total_negative_weight

  tpr = cumulative_positive_weights / cumulative_positive_weights[-1]
  fpr = cumulative_negative_weights / cumulative_negative_weights[-1]

  # Calculate AUC
  auc_value = np.trapz(tpr, fpr)

  dict[name] = auc_value

  # Plot ROC curve
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Weighted ROC curve (AUC = {auc_value:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic with Weights')
  plt.legend(loc="lower right")
  plt.show()
#%%
with open(f'{signal_name_}_AUC_Threshold.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, dict.keys())
    w.writeheader()
    w.writerow(dict)
