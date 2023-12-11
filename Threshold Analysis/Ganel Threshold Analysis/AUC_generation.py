#Load packages
import pandas as pd
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import csv
from alive_progress import alive_bar
import time

#Sets style for matplotlib
hep.style.use("CMS")

#Loads in the dataset
df = pd.read_parquet("merged_nominal.parquet")

#Loads in the process names
with open("summary.json", "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

  #Creates the background and signal datasets.
background = pd.concat((df[df.process_id == proc_dict["DiPhoton"]],df[df.process_id == proc_dict["TTGG"]],df[df.process_id == proc_dict["TTGamma"]],
                        df[df.process_id == proc_dict["TTJets"]],df[df.process_id == proc_dict["VBFH_M125"]],df[df.process_id == proc_dict["WGamma"]],
                        df[df.process_id == proc_dict["ZGamma"]],df[df.process_id == proc_dict["ggH_M125"]],df[df.process_id == proc_dict["ttH_M125"]],
                        df[df.process_id == proc_dict["GJets"]]), ignore_index=True, axis=0)

signal = df[df.process_id == proc_dict["GluGluToRadionToHHTo2G2Tau_M-400"]]

feature_info = pd.read_csv("ROC_feature_info.csv")

print(feature_info.head())
print(feature_info.loc[feature_info['Feature'] == 'reco_MX_mgg'])
print(feature_info.loc[feature_info['Feature'] == 'reco_MX_mgg'])

name = 'SubleadPhoton_pt_mgg'

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
    print(name)
    feature_info_row = feature_info.loc[feature_info['Feature'] == name]


    hist_signal, bin_edges_signal = np.histogram(signal[name], bins=80, range=(feature_info_row['Range_Low'].item(), feature_info_row['Range_High'].item()))
    hist_background, bin_edges_background = np.histogram(background[name], bins=80, range=(feature_info_row['Range_Low'].item(), feature_info_row['Range_High'].item()))

    #Gets the maximum and minimum edge values
    minimum_edge = min(min(bin_edges_signal),min(bin_edges_background))
    maximum_edge = max(max(bin_edges_signal),max(bin_edges_background))

    signal_ = signal.loc[(signal[name] >= feature_info_row['Range_Low'].item())]
    background_ = background.loc[(background[name] >= feature_info_row['Range_Low'].item())]

    #step size for each threshold value (100 steps)
    step_number = 1000
    step_size = (maximum_edge - minimum_edge)/step_number

    TPR_arr = []
    FPR_arr = []
    step_size_arr = []

    i = minimum_edge
    count = 1

    if str(feature_info_row['Signal'].item()) == 'u':
        TP = sum(signal_[signal_[name] >= i].weight_central)
        FP = sum(background_[background_[name] >= i].weight_central)
        FN = sum(signal_[signal_[name] < i].weight_central)
        TN = sum(background_[background_[name] < i].weight_central)

        TPR_normalise = (TP +FN)

        FPR_normalise = (FP +TN)
        
        with alive_bar(1000) as bar:
            while i < (maximum_edge+step_size):
                #print(f'{count}/{int(((maximum_edge-minimum_edge)+step_size)/step_size)}')
                count += 1
                i += step_size
                TP = sum(signal_[signal_[name] >= i].weight_central)
                FP = sum(background_[background_[name] >= i].weight_central)

                TPR = TP/TPR_normalise
                FPR = FP/FPR_normalise
                TPR_arr.append(TPR)
                FPR_arr.append(FPR)
                step_size_arr.append(i)
                bar()
    else:
        TP = sum(signal_[signal_[name] <= i].weight_central)
        FP = sum(background_[background_[name] <= i].weight_central)
        FN = sum(signal_[signal_[name] > i].weight_central)
        TN = sum(background_[background_[name] > i].weight_central)

        TPR_normalise = (TP +FN)

        FPR_normalise = (FP +TN)

        with alive_bar(1000) as bar:
            while i < (maximum_edge+step_size):
                #print(f'{count}/{int(((maximum_edge-minimum_edge)+step_size)/step_size)}')
                count += 1
                i += step_size
                TP = sum(signal_[signal_[name] <= i].weight_central)
                FP = sum(background_[background_[name] <= i].weight_central)

                TPR = TP/TPR_normalise
                FPR = FP/FPR_normalise
                TPR_arr.append(TPR)
                FPR_arr.append(FPR)
                step_size_arr.append(i)
                bar()
            


    roc_auc = abs(np.trapz(TPR_arr,FPR_arr,dx =step_size))
    fig = plt.figure()
    plt.scatter(FPR_arr,TPR_arr, c = 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(f'ROC Curve for {name}')
    plt.legend()
    plt.savefig(f"ROC_Curves_Threshold_Central/m_400/{name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"ROC_Curves_Threshold_Central/m_400/{name}.jpg", format="jpg", bbox_inches="tight")

    dict[name] = roc_auc

    fig = plt.figure()
    diff = np.array(TPR_arr)-np.array(FPR_arr)
    max_value = max(diff)
    index_ = np.where(diff == max_value)[0][0]
    plt.scatter(step_size_arr,diff, label = f'Maximum Value: {step_size_arr[index_]:4g}')
    plt.xlabel('Threshold Value')
    plt.ylabel('TPR - FPR')
    plt.legend()
    plt.title(f'Threshold Number vs (TPR - FPR) for {name}')
    plt.savefig(f"TPR_FPR_Threshold_Best_Value_Graphs_Central/m_400/{name}.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"TPR_FPR_Threshold_Best_Value_Graphs_Central/m_400/{name}.jpg", format="jpg", bbox_inches="tight")

with open('GluGluToRadionToHHTo2G2Tau_M_400_AUC.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, dict.keys())
    w.writeheader()
    w.writerow(dict)

