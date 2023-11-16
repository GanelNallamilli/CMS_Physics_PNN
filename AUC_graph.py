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

feature_list = ['Diphoton_mass', 'Diphoton_pt_mgg', 'Diphoton_dPhi',
       'LeadPhoton_pt_mgg', 'SubleadPhoton_pt_mgg', 'MET_pt',
       'diphoton_met_dPhi', 'ditau_met_dPhi', 'ditau_deta', 'lead_lepton_pt',
       'lead_lepton_mass', 'jet_1_pt', 'ditau_pt', 'ditau_mass',
       'ditau_dR', 'ditau_dphi', 'Diphoton_ditau_dphi', 'dilep_leadpho_mass','reco_MX_mgg',
       'Diphoton_ditau_deta', 'Diphoton_lead_lepton_deta',
       'Diphoton_lead_lepton_dR', 'Diphoton_sublead_lepton_deta',
       'Diphoton_sublead_lepton_dR', 'LeadPhoton_ditau_dR',
       'LeadPhoton_lead_lepton_dR', 'SubleadPhoton_lead_lepton_dR']

GluGluToRadionToHHTo2G2Tau_M_300_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_300_AUC.csv")
GluGluToRadionToHHTo2G2Tau_M_400_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_400_AUC.csv")
GluGluToRadionToHHTo2G2Tau_M_600_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_600_AUC.csv")
GluGluToRadionToHHTo2G2Tau_M_900_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_900_AUC.csv")
GluGluToRadionToHHTo2G2Tau_M_800_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_800_AUC.csv")
GluGluToRadionToHHTo2G2Tau_M_1000_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_1000_AUC.csv")

print(GluGluToRadionToHHTo2G2Tau_M_300_AUC.head())

GluGluToRadionToHHTo2G2Tau_M_300_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_300_AUC.iloc[0]).to_numpy()
GluGluToRadionToHHTo2G2Tau_M_400_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_400_AUC.iloc[0]).to_numpy()
GluGluToRadionToHHTo2G2Tau_M_600_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_600_AUC.iloc[0]).to_numpy()
GluGluToRadionToHHTo2G2Tau_M_900_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_900_AUC.iloc[0]).to_numpy()
GluGluToRadionToHHTo2G2Tau_M_800_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_800_AUC.iloc[0]).to_numpy()
GluGluToRadionToHHTo2G2Tau_M_1000_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_1000_AUC.iloc[0]).to_numpy()

print(GluGluToRadionToHHTo2G2Tau_M_300_AUC_np)

plt.xticks(ticks=range(len(GluGluToRadionToHHTo2G2Tau_M_300_AUC_np)), labels=feature_list, rotation=90)
plt.plot(GluGluToRadionToHHTo2G2Tau_M_300_AUC_np, marker = 'x', label = 'GluGluToRadionToHHTo2G2Tau_M_300_AUC')
plt.plot(GluGluToRadionToHHTo2G2Tau_M_400_AUC_np, marker = 'x', label = 'GluGluToRadionToHHTo2G2Tau_M_400_AUC')
plt.plot(GluGluToRadionToHHTo2G2Tau_M_600_AUC_np, marker = 'x', label = 'GluGluToRadionToHHTo2G2Tau_M_600_AUC')
plt.plot(GluGluToRadionToHHTo2G2Tau_M_800_AUC_np, marker = 'x', label = 'GluGluToRadionToHHTo2G2Tau_M_800_AUC')
plt.plot(GluGluToRadionToHHTo2G2Tau_M_900_AUC_np, marker = 'x', label = 'GluGluToRadionToHHTo2G2Tau_M_900_AUC')
plt.plot(GluGluToRadionToHHTo2G2Tau_M_1000_AUC_np, marker = 'x', label = 'GluGluToRadionToHHTo2G2Tau_M_1000_AUC')
plt.legend()
plt.show()
