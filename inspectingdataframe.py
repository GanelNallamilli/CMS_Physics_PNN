# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 13:24:24 2023

@author: drpla
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
hep.style.use("CMS")
import scipy as scipy

df = pd.read_parquet(r'C:\Users\drpla\Desktop\ICL-PHYSICS-YEAR-4\Masters Project\Data\New folder\merged_nominal.parquet')
#%%
with open(r'C:\Users\drpla\Desktop\ICL-PHYSICS-YEAR-4\Masters Project\Data\New folder\summary.json', "r") as f:
  proc_dict = json.load(f)["sample_id_map"]
  
  
sig = df[df.process_id == proc_dict["GluGluToRadionToHHTo2G2Tau_M-250"]] # just one signal process, mass of X is 1000 GeV
bkg = df[df.process_id == proc_dict["DiPhoton"]] # just one of the background processes

plt.hist(sig.Diphoton_mass, range=(100, 180), bins=80, histtype="step", label="Signal")
plt.hist(bkg.Diphoton_mass, range=(100, 180), bins=80, histtype="step", label="Background")
plt.legend()
plt.xlabel(r"$m_{\gamma\gamma}$")
plt.show()
#%%
def feature_name(index):
    print(df.columns[index])
#%%

feature_index=0
sig = df[df.process_id == proc_dict["GJets"]] #GluGluToRadionToHHTo2G2Tau_M-1000# just one signal process, mass of X is 1000 GeV
bkg = df[df.process_id == proc_dict["DiPhoton"]] # just one of the background processes
i=0
for column_name in sig.columns:

        sigg=sig[column_name]
        bkgg=bkg[column_name]
        plt.hist(sigg, range=(100, 180), bins=80, histtype="step", label=column_name)
        #plt.hist(bkgg, range=(100, 180), bins=80, histtype="step", label="Background")
        #plt.legend()
        plt.xlabel(r"$m_{\gamma\gamma}$")
        plt.show()
        i+=1
        if i > 100:
            break


#%%

"""
Plotting the histograms for features
"""
signal_list=['GluGluToRadionToHHTo2G2Tau_M-1000','GluGluToRadionToHHTo2G2Tau_M-250',
 'GluGluToRadionToHHTo2G2Tau_M-260',
 'GluGluToRadionToHHTo2G2Tau_M-270',
 'GluGluToRadionToHHTo2G2Tau_M-280',
 'GluGluToRadionToHHTo2G2Tau_M-290',
 'GluGluToRadionToHHTo2G2Tau_M-300',
 'GluGluToRadionToHHTo2G2Tau_M-320',
 'GluGluToRadionToHHTo2G2Tau_M-350',
 'GluGluToRadionToHHTo2G2Tau_M-400',
 'GluGluToRadionToHHTo2G2Tau_M-450',
 'GluGluToRadionToHHTo2G2Tau_M-500',
 'GluGluToRadionToHHTo2G2Tau_M-550',
 'GluGluToRadionToHHTo2G2Tau_M-600',
 'GluGluToRadionToHHTo2G2Tau_M-650',
 'GluGluToRadionToHHTo2G2Tau_M-700',
 'GluGluToRadionToHHTo2G2Tau_M-750',
 'GluGluToRadionToHHTo2G2Tau_M-800',
 'GluGluToRadionToHHTo2G2Tau_M-900']
signal_list=['ggH_M125']
# =============================================================================
# signal_list=['Data','DiPhoton', 'TTGG', 'TTGamma',
#  'TTJets',
#  'VBFH_M125',
#  'VH_M125',
#  'WGamma',
#  'ZGamma',
#  'ggH_M125',
#  'ttH_M125',
#  'GJets']
# =============================================================================
event_features=['Diphoton_mass','Diphoton_pt_mgg','Diphoton_dPhi','LeadPhoton_pt_mgg','SubleadPhoton_pt_mgg','MET_pt','diphoton_met_dPhi','ditau_met_dPhi','ditau_deta','lead_lepton_pt','lead_lepton_mass','category','jet_1_pt','ditau_pt','ditau_mass','ditau_dR','ditau_dphi','Diphoton_ditau_dphi','dilep_leadpho_mass','event','process_id','year','MX','MY','reco_MX_mgg','Diphoton_ditau_deta','Diphoton_lead_lepton_deta','Diphoton_lead_lepton_dR','Diphoton_sublead_lepton_deta','Diphoton_sublead_lepton_dR','LeadPhoton_ditau_dR','LeadPhoton_lead_lepton_dR','SubleadPhoton_lead_lepton_dR','weight_central','weight_photon_presel_sf_Diphoton_Photon_up','weight_central_initial','weight_btag_deepjet_sf_SelectedJet_up_lfstats1','weight_btag_deepjet_sf_SelectedJet_up_cferr1','weight_muon_id_sfSYS_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_central','weight_photon_id_sf_Diphoton_Photon_up','weight_photon_presel_sf_Diphoton_Photon_down','weight_electron_veto_sf_Diphoton_Photon_up','weight_muon_iso_sfSTAT_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_down','weight_L1_prefiring_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats2','weight_muon_id_sfSTAT_SelectedMuon_up','weight_btag_deepjet_sf_SelectedJet_up_jes','weight_trigger_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats1','weight_muon_iso_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSYS_SelectedMuon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr1','weight_photon_id_sf_Diphoton_Photon_central','weight_muon_id_sfSYS_SelectedMuon_up','weight_electron_id_sf_SelectedElectron_up','weight_muon_id_sfSYS_SelectedMuon_central','weight_tau_idDeepTauVSe_sf_AnalysisTau_down','weight_puWeight_central','weight_btag_deepjet_sf_SelectedJet_down_lfstats2','weight_tau_idDeepTauVSe_sf_AnalysisTau_central','weight_tau_idDeepTauVSjet_sf_AnalysisTau_down','weight_trigger_sf_central','weight_photon_presel_sf_Diphoton_Photon_central','weight_electron_id_sf_SelectedElectron_down','weight_btag_deepjet_sf_SelectedJet_down_lf','weight_puWeight_up','weight_btag_deepjet_sf_SelectedJet_up_hfstats2','weight_btag_deepjet_sf_SelectedJet_down_lfstats1','weight_puWeight_down','weight_muon_iso_sfSYS_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_up_hfstats1','weight_tau_idDeepTauVSjet_sf_AnalysisTau_up','weight_electron_veto_sf_Diphoton_Photon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr2','weight_L1_prefiring_sf_down','weight_muon_id_sfSTAT_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_down_jes','weight_btag_deepjet_sf_SelectedJet_down_hf','weight_electron_veto_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_central','weight_btag_deepjet_sf_SelectedJet_up_hf','weight_electron_id_sf_SelectedElectron_central','weight_trigger_sf_down','weight_tau_idDeepTauVSe_sf_AnalysisTau_up','weight_tau_idDeepTauVSmu_sf_AnalysisTau_up','weight_photon_id_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_up_lf','weight_muon_id_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSTAT_SelectedMuon_up','weight_central_no_lumi','weight_btag_deepjet_sf_SelectedJet_up_cferr2','weight_btag_deepjet_sf_SelectedJet_up_lfstats2','weight_muon_iso_sfSYS_SelectedMuon_up','weight_tau_idDeepTauVSjet_sf_AnalysisTau_central','weight_L1_prefiring_sf_central']
j=33
event_features_index=j

i=1
for num, signal in enumerate(signal_list):
    sig1=df[df.process_id == proc_dict[signal]]
    sig1=sig1[event_features[event_features_index]] 
    plt.hist(sig1, bins=80, histtype="step", label=signal_list[num])
    i+=1
plt.legend()
#plt.xlabel(r"$m_{\gamma\gamma}$")
plt.title(event_features[j])
plt.show()
print("Histograms for all signals for",j,':',  event_features[j])
j+=1
if j ==97:
    j=0


 #%%



"""
Plotting AUC and ROC of the background and signal classifier.
"""

hist_range=None #def histogram range
feature='Diphoton_pt_mgg' #choose feature to plot

signal_list=['GluGluToRadionToHHTo2G2Tau_M-1000', #list of all GluGlu signals
             'GluGluToRadionToHHTo2G2Tau_M-250',
 'GluGluToRadionToHHTo2G2Tau_M-260',
 'GluGluToRadionToHHTo2G2Tau_M-270',
 'GluGluToRadionToHHTo2G2Tau_M-280',
 'GluGluToRadionToHHTo2G2Tau_M-290',
 'GluGluToRadionToHHTo2G2Tau_M-300',
 'GluGluToRadionToHHTo2G2Tau_M-320',
 'GluGluToRadionToHHTo2G2Tau_M-350',
 'GluGluToRadionToHHTo2G2Tau_M-400',
 'GluGluToRadionToHHTo2G2Tau_M-450',
 'GluGluToRadionToHHTo2G2Tau_M-500',
 'GluGluToRadionToHHTo2G2Tau_M-550',
 'GluGluToRadionToHHTo2G2Tau_M-600',
 'GluGluToRadionToHHTo2G2Tau_M-650',
 'GluGluToRadionToHHTo2G2Tau_M-700',
 'GluGluToRadionToHHTo2G2Tau_M-750',
 'GluGluToRadionToHHTo2G2Tau_M-800',
 'GluGluToRadionToHHTo2G2Tau_M-900']

signal='GluGluToRadionToHHTo2G2Tau_M_900' # list of one specific signal 

event_features=['Diphoton_mass', # list of all event features to choose from
                'Diphoton_pt_mgg','Diphoton_dPhi','LeadPhoton_pt_mgg','SubleadPhoton_pt_mgg','MET_pt','diphoton_met_dPhi','ditau_met_dPhi','ditau_deta','lead_lepton_pt','lead_lepton_mass','category','jet_1_pt','ditau_pt','ditau_mass','ditau_dR','ditau_dphi','Diphoton_ditau_dphi','dilep_leadpho_mass','event','process_id','year','MX','MY','reco_MX_mgg','Diphoton_ditau_deta','Diphoton_lead_lepton_deta','Diphoton_lead_lepton_dR','Diphoton_sublead_lepton_deta','Diphoton_sublead_lepton_dR','LeadPhoton_ditau_dR','LeadPhoton_lead_lepton_dR','SubleadPhoton_lead_lepton_dR','weight_central','weight_photon_presel_sf_Diphoton_Photon_up','weight_central_initial','weight_btag_deepjet_sf_SelectedJet_up_lfstats1','weight_btag_deepjet_sf_SelectedJet_up_cferr1','weight_muon_id_sfSYS_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_central','weight_photon_id_sf_Diphoton_Photon_up','weight_photon_presel_sf_Diphoton_Photon_down','weight_electron_veto_sf_Diphoton_Photon_up','weight_muon_iso_sfSTAT_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_down','weight_L1_prefiring_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats2','weight_muon_id_sfSTAT_SelectedMuon_up','weight_btag_deepjet_sf_SelectedJet_up_jes','weight_trigger_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats1','weight_muon_iso_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSYS_SelectedMuon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr1','weight_photon_id_sf_Diphoton_Photon_central','weight_muon_id_sfSYS_SelectedMuon_up','weight_electron_id_sf_SelectedElectron_up','weight_muon_id_sfSYS_SelectedMuon_central','weight_tau_idDeepTauVSe_sf_AnalysisTau_down','weight_puWeight_central','weight_btag_deepjet_sf_SelectedJet_down_lfstats2','weight_tau_idDeepTauVSe_sf_AnalysisTau_central','weight_tau_idDeepTauVSjet_sf_AnalysisTau_down','weight_trigger_sf_central','weight_photon_presel_sf_Diphoton_Photon_central','weight_electron_id_sf_SelectedElectron_down','weight_btag_deepjet_sf_SelectedJet_down_lf','weight_puWeight_up','weight_btag_deepjet_sf_SelectedJet_up_hfstats2','weight_btag_deepjet_sf_SelectedJet_down_lfstats1','weight_puWeight_down','weight_muon_iso_sfSYS_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_up_hfstats1','weight_tau_idDeepTauVSjet_sf_AnalysisTau_up','weight_electron_veto_sf_Diphoton_Photon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr2','weight_L1_prefiring_sf_down','weight_muon_id_sfSTAT_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_down_jes','weight_btag_deepjet_sf_SelectedJet_down_hf','weight_electron_veto_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_central','weight_btag_deepjet_sf_SelectedJet_up_hf','weight_electron_id_sf_SelectedElectron_central','weight_trigger_sf_down','weight_tau_idDeepTauVSe_sf_AnalysisTau_up','weight_tau_idDeepTauVSmu_sf_AnalysisTau_up','weight_photon_id_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_up_lf','weight_muon_id_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSTAT_SelectedMuon_up','weight_central_no_lumi','weight_btag_deepjet_sf_SelectedJet_up_cferr2','weight_btag_deepjet_sf_SelectedJet_up_lfstats2','weight_muon_iso_sfSYS_SelectedMuon_up','weight_tau_idDeepTauVSjet_sf_AnalysisTau_central','weight_L1_prefiring_sf_central']



background_list=['Data','DiPhoton', 'TTGG', 'TTGamma',#list of each bkgs for concatenation
 'TTJets',
 'VBFH_M125',
 'VH_M125',
 'WGamma',
 'ZGamma',
 'ggH_M125', 
 'ttH_M125',
 'GJets']

listforconc=[]
for i in background_list:                               #creating a concatenated list of bkg
    bkgg = df[df.process_id == proc_dict[i]][feature]
    listforconc.append(bkgg)
    
background = pd.concat(listforconc)

plt.hist(background, range=hist_range, bins=80, histtype="step", label='Concatenated Background')
plt.title(feature)
sig = df[df.process_id == proc_dict["GluGluToRadionToHHTo2G2Tau_M-1000"]] # just one signal process, mass of X is 1000 GeV

plt.hist(sig[feature], range=hist_range, bins=80, histtype="step", label=("Signal ", signal))
plt.xlabel(r"$m_{\gamma\gamma}$")
plt.legend()
"""
#to plot a non-concatenated background dataframe as comparison

bkg = df[df.process_id == proc_dict["DiPhoton"]] # just one of the background processes
plt.hist(bkg[feature], range=None, bins=80, histtype="step", label="non-conc Background")
"""

sig_classification=list(np.ones(sig[feature].size,dtype=int))
sig_classifying=pd.DataFrame(sig[feature])
sig_classifying["Classification"]=sig_classification

bkg_classification=list(np.zeros(background.size,dtype=int))
bkg_classifying=pd.DataFrame(background)
bkg_classifying["Classification"]=bkg_classification

classified = pd.concat([sig_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)

classified['Cumulative Signal'] = (((classified['Classification'] == 1))/(sig[feature].size)).cumsum()
classified['Cumulative Background'] = (((classified['Classification'] == 0))/(background.size)).cumsum()

#%%
"""
plot for above
"""
plt.plot(classified['Cumulative Signal'],classified['Cumulative Background'])
plt.ylabel('Sensitivity')
plt.xlabel('FPR')
plt.title(feature)

plt.legend()
plt.show()

#%%
# =============================================================================
# 
# 
# i=1
# for num, signal in enumerate(background_list):
#     sig1=df[df.process_id == proc_dict[signal]]
#     sig1=sig1[event_features[event_features_index]]
#     plt.hist(sig1, bins=80, histtype="step", label=signal_list[num])
#     i+=1
# #plt.legend()
# plt.xlabel(r"$m_{\gamma\gamma}$")
# plt.title(event_features[j])
# plt.show()
# print("Histograms for all signals for",j,':',  event_features[j])
# j+=1
# if j ==97:
#     j=0
# 
# 
# #%%
# 
# plt.hist(background, range=hist_range, bins=80, histtype="step", label='Concatenated Background')
# plt.hist(df[df.process_id == proc_dict["GluGluToRadionToHHTo2G2Tau_M-900"]], range=hist_range, bins=80, histtype="step", label=("Signal ", signal))
# 
# =============================================================================


#%%
hist_range=(80,180) #def histogram range
hist_range=None
feature='ditau_deta' #choose feature to plot


signal_list=['GluGluToRadionToHHTo2G2Tau_M-1000', #list of all GluGlu signals
             'GluGluToRadionToHHTo2G2Tau_M-250',
 'GluGluToRadionToHHTo2G2Tau_M-260',
 'GluGluToRadionToHHTo2G2Tau_M-270',
 'GluGluToRadionToHHTo2G2Tau_M-280',
 'GluGluToRadionToHHTo2G2Tau_M-290',
 'GluGluToRadionToHHTo2G2Tau_M-300',
 'GluGluToRadionToHHTo2G2Tau_M-320',
 'GluGluToRadionToHHTo2G2Tau_M-350',
 'GluGluToRadionToHHTo2G2Tau_M-400',
 'GluGluToRadionToHHTo2G2Tau_M-450',
 'GluGluToRadionToHHTo2G2Tau_M-500',
 'GluGluToRadionToHHTo2G2Tau_M-550',
 'GluGluToRadionToHHTo2G2Tau_M-600',
 'GluGluToRadionToHHTo2G2Tau_M-650',
 'GluGluToRadionToHHTo2G2Tau_M-700',
 'GluGluToRadionToHHTo2G2Tau_M-750',
 'GluGluToRadionToHHTo2G2Tau_M-800',
 'GluGluToRadionToHHTo2G2Tau_M-900']

signal='GluGluToRadionToHHTo2G2Tau_M-900' # list of one specific signal 

event_features=['Diphoton_mass', # list of all event features to choose from
                'Diphoton_pt_mgg','Diphoton_dPhi','LeadPhoton_pt_mgg','SubleadPhoton_pt_mgg','MET_pt','diphoton_met_dPhi','ditau_met_dPhi','ditau_deta','lead_lepton_pt','lead_lepton_mass','category','jet_1_pt','ditau_pt','ditau_mass','ditau_dR','ditau_dphi','Diphoton_ditau_dphi','dilep_leadpho_mass','event','process_id','year','MX','MY','reco_MX_mgg','Diphoton_ditau_deta','Diphoton_lead_lepton_deta','Diphoton_lead_lepton_dR','Diphoton_sublead_lepton_deta','Diphoton_sublead_lepton_dR','LeadPhoton_ditau_dR','LeadPhoton_lead_lepton_dR','SubleadPhoton_lead_lepton_dR','weight_central','weight_photon_presel_sf_Diphoton_Photon_up','weight_central_initial','weight_btag_deepjet_sf_SelectedJet_up_lfstats1','weight_btag_deepjet_sf_SelectedJet_up_cferr1','weight_muon_id_sfSYS_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_central','weight_photon_id_sf_Diphoton_Photon_up','weight_photon_presel_sf_Diphoton_Photon_down','weight_electron_veto_sf_Diphoton_Photon_up','weight_muon_iso_sfSTAT_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_down','weight_L1_prefiring_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats2','weight_muon_id_sfSTAT_SelectedMuon_up','weight_btag_deepjet_sf_SelectedJet_up_jes','weight_trigger_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats1','weight_muon_iso_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSYS_SelectedMuon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr1','weight_photon_id_sf_Diphoton_Photon_central','weight_muon_id_sfSYS_SelectedMuon_up','weight_electron_id_sf_SelectedElectron_up','weight_muon_id_sfSYS_SelectedMuon_central','weight_tau_idDeepTauVSe_sf_AnalysisTau_down','weight_puWeight_central','weight_btag_deepjet_sf_SelectedJet_down_lfstats2','weight_tau_idDeepTauVSe_sf_AnalysisTau_central','weight_tau_idDeepTauVSjet_sf_AnalysisTau_down','weight_trigger_sf_central','weight_photon_presel_sf_Diphoton_Photon_central','weight_electron_id_sf_SelectedElectron_down','weight_btag_deepjet_sf_SelectedJet_down_lf','weight_puWeight_up','weight_btag_deepjet_sf_SelectedJet_up_hfstats2','weight_btag_deepjet_sf_SelectedJet_down_lfstats1','weight_puWeight_down','weight_muon_iso_sfSYS_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_up_hfstats1','weight_tau_idDeepTauVSjet_sf_AnalysisTau_up','weight_electron_veto_sf_Diphoton_Photon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr2','weight_L1_prefiring_sf_down','weight_muon_id_sfSTAT_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_down_jes','weight_btag_deepjet_sf_SelectedJet_down_hf','weight_electron_veto_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_central','weight_btag_deepjet_sf_SelectedJet_up_hf','weight_electron_id_sf_SelectedElectron_central','weight_trigger_sf_down','weight_tau_idDeepTauVSe_sf_AnalysisTau_up','weight_tau_idDeepTauVSmu_sf_AnalysisTau_up','weight_photon_id_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_up_lf','weight_muon_id_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSTAT_SelectedMuon_up','weight_central_no_lumi','weight_btag_deepjet_sf_SelectedJet_up_cferr2','weight_btag_deepjet_sf_SelectedJet_up_lfstats2','weight_muon_iso_sfSYS_SelectedMuon_up','weight_tau_idDeepTauVSjet_sf_AnalysisTau_central','weight_L1_prefiring_sf_central']



background_list=['Data','DiPhoton', 'TTGG', 'TTGamma',#list of each bkgs for concatenation
 'TTJets',
 'VBFH_M125',
 'VH_M125',
 'WGamma',
 'ZGamma',
 'ggH_M125', 
 'ttH_M125',
 'GJets']

sig = df[df.process_id == proc_dict[signal]]

fig=plt.figure()

plt.hist(background[feature], range=hist_range, bins=80, histtype="step", label='Concatenated Background')
plt.hist(sig_classifying[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal}"))
plt.legend(fontsize="15")
plt.show()

#%%


listforconc=[]
for i in background_list:                           #creating a concatenated list of bkg

    bkgg = df[df.process_id == proc_dict[i]]#[feature]
    listforconc.append(bkgg)
    
background = pd.concat(listforconc)


sig_classification=list(np.ones(sig[feature].size,dtype=int))
sig_classifying=pd.DataFrame(sig)
sig_classifying["Classification"]=sig_classification

bkg_classification=list(np.zeros(background[feature].size,dtype=int))
bkg_classifying=pd.DataFrame(background)
bkg_classifying["Classification"]=bkg_classification



sig_bkg=pd.concat([sig_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)



minimum_edge=min(sig_bkg[feature])
maximum_edge=max(sig_bkg[feature])
steps=100
stepsize=(maximum_edge-minimum_edge)/steps
#%%
direction='below' #use this to determine the side of the threshold that calculates TPR and FPR

threshold_direction = f'Signal is {direction} Background'
threshold=minimum_edge
TPR_arr = []
FPR_arr = []
for i in range(0,steps): 
    if threshold_direction == 'Signal is above Background':
        threshold += stepsize
       
        TP = len(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==1)][feature])#
        FP = len(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==0)][feature])
        FN = len(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==1)][feature])
        TN = len(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==0)][feature])
        
        """
        TP = sum(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        FP = sum(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        TN = sum(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        FN = sum(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        """
        print('run',i)
        
        TPR = TP/(TP +FN)
        FPR = FP/(FP +TN)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)
        
    elif threshold_direction == 'Signal is below Background':
        threshold += stepsize
       
        TP = len(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==1)][feature])#
        FP = len(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==0)][feature])
        FN = len(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==1)][feature])
        TN = len(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==0)][feature])
        
        """
        TP = sum(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        FP = sum(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        TN = sum(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        FN = sum(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        """
        print('run',i)
        
        TPR = TP/(TP +FN)
        FPR = FP/(FP +TN)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)
        


sorted_values = sorted(zip(FPR_arr, TPR_arr))
FPR_arr, TPR_arr = zip(*sorted_values)

AUCscore=scipy.integrate.trapezoid(TPR_arr,FPR_arr,  dx=stepsize,axis=-1)
fig=plt.figure()    
plt.scatter(FPR_arr,TPR_arr,label=f'AUC:{AUCscore}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
print(AUCscore)

#%%
def trapezoidal_rule(x, y):
    """
    Compute the definite integral of the discrete data points given by
    'x' (list of x values) and 'y' (list of corresponding y values)
    using the trapezoidal rule.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Lists x and y must have the same length")

    integral = 0.0
    for i in range(1, n):
        # Calculate the width of each interval
        h = x[i] - x[i - 1]

        # Use trapezoidal rule formula for each interval and sum them up
        integral += 0.5 * (y[i] + y[i - 1]) * h
        print(integral)
    return integral

AUC=trapezoidal_rule(FPR_arr,TPR_arr)



#
hist_range=(80,180) #def histogram range
hist_range=None
feature='ditau_deta' #choose feature to plot


signal_list=['GluGluToRadionToHHTo2G2Tau_M-1000', #list of all GluGlu signals
             'GluGluToRadionToHHTo2G2Tau_M-250',
 'GluGluToRadionToHHTo2G2Tau_M-260',
 'GluGluToRadionToHHTo2G2Tau_M-270',
 'GluGluToRadionToHHTo2G2Tau_M-280',
 'GluGluToRadionToHHTo2G2Tau_M-290',
 'GluGluToRadionToHHTo2G2Tau_M-300',
 'GluGluToRadionToHHTo2G2Tau_M-320',
 'GluGluToRadionToHHTo2G2Tau_M-350',
 'GluGluToRadionToHHTo2G2Tau_M-400',
 'GluGluToRadionToHHTo2G2Tau_M-450',
 'GluGluToRadionToHHTo2G2Tau_M-500',
 'GluGluToRadionToHHTo2G2Tau_M-550',
 'GluGluToRadionToHHTo2G2Tau_M-600',
 'GluGluToRadionToHHTo2G2Tau_M-650',
 'GluGluToRadionToHHTo2G2Tau_M-700',
 'GluGluToRadionToHHTo2G2Tau_M-750',
 'GluGluToRadionToHHTo2G2Tau_M-800',
 'GluGluToRadionToHHTo2G2Tau_M-900']

signal='GluGluToRadionToHHTo2G2Tau_M-900' # list of one specific signal 

event_features=['Diphoton_mass', # list of all event features to choose from
                'Diphoton_pt_mgg','Diphoton_dPhi','LeadPhoton_pt_mgg','SubleadPhoton_pt_mgg','MET_pt','diphoton_met_dPhi','ditau_met_dPhi','ditau_deta','lead_lepton_pt','lead_lepton_mass','category','jet_1_pt','ditau_pt','ditau_mass','ditau_dR','ditau_dphi','Diphoton_ditau_dphi','dilep_leadpho_mass','event','process_id','year','MX','MY','reco_MX_mgg','Diphoton_ditau_deta','Diphoton_lead_lepton_deta','Diphoton_lead_lepton_dR','Diphoton_sublead_lepton_deta','Diphoton_sublead_lepton_dR','LeadPhoton_ditau_dR','LeadPhoton_lead_lepton_dR','SubleadPhoton_lead_lepton_dR','weight_central','weight_photon_presel_sf_Diphoton_Photon_up','weight_central_initial','weight_btag_deepjet_sf_SelectedJet_up_lfstats1','weight_btag_deepjet_sf_SelectedJet_up_cferr1','weight_muon_id_sfSYS_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_central','weight_photon_id_sf_Diphoton_Photon_up','weight_photon_presel_sf_Diphoton_Photon_down','weight_electron_veto_sf_Diphoton_Photon_up','weight_muon_iso_sfSTAT_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_down','weight_L1_prefiring_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats2','weight_muon_id_sfSTAT_SelectedMuon_up','weight_btag_deepjet_sf_SelectedJet_up_jes','weight_trigger_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats1','weight_muon_iso_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSYS_SelectedMuon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr1','weight_photon_id_sf_Diphoton_Photon_central','weight_muon_id_sfSYS_SelectedMuon_up','weight_electron_id_sf_SelectedElectron_up','weight_muon_id_sfSYS_SelectedMuon_central','weight_tau_idDeepTauVSe_sf_AnalysisTau_down','weight_puWeight_central','weight_btag_deepjet_sf_SelectedJet_down_lfstats2','weight_tau_idDeepTauVSe_sf_AnalysisTau_central','weight_tau_idDeepTauVSjet_sf_AnalysisTau_down','weight_trigger_sf_central','weight_photon_presel_sf_Diphoton_Photon_central','weight_electron_id_sf_SelectedElectron_down','weight_btag_deepjet_sf_SelectedJet_down_lf','weight_puWeight_up','weight_btag_deepjet_sf_SelectedJet_up_hfstats2','weight_btag_deepjet_sf_SelectedJet_down_lfstats1','weight_puWeight_down','weight_muon_iso_sfSYS_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_up_hfstats1','weight_tau_idDeepTauVSjet_sf_AnalysisTau_up','weight_electron_veto_sf_Diphoton_Photon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr2','weight_L1_prefiring_sf_down','weight_muon_id_sfSTAT_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_down_jes','weight_btag_deepjet_sf_SelectedJet_down_hf','weight_electron_veto_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_central','weight_btag_deepjet_sf_SelectedJet_up_hf','weight_electron_id_sf_SelectedElectron_central','weight_trigger_sf_down','weight_tau_idDeepTauVSe_sf_AnalysisTau_up','weight_tau_idDeepTauVSmu_sf_AnalysisTau_up','weight_photon_id_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_up_lf','weight_muon_id_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSTAT_SelectedMuon_up','weight_central_no_lumi','weight_btag_deepjet_sf_SelectedJet_up_cferr2','weight_btag_deepjet_sf_SelectedJet_up_lfstats2','weight_muon_iso_sfSYS_SelectedMuon_up','weight_tau_idDeepTauVSjet_sf_AnalysisTau_central','weight_L1_prefiring_sf_central']



background_list=['Data','DiPhoton', 'TTGG', 'TTGamma',#list of each bkgs for concatenation
 'TTJets',
 'VBFH_M125',
 'VH_M125',
 'WGamma',
 'ZGamma',
 'ggH_M125', 
 'ttH_M125',
 'GJets']

sig = df[df.process_id == proc_dict[signal]]

fig=plt.figure()

plt.hist(background[feature], range=hist_range, bins=80, histtype="step", label='Concatenated Background')
plt.hist(sig_classifying[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal}"))
plt.legend(fontsize="15")
plt.show()

#%%


listforconc=[]
for i in background_list:                           #creating a concatenated list of bkg

    bkgg = df[df.process_id == proc_dict[i]]#[feature]
    listforconc.append(bkgg)
    
background = pd.concat(listforconc)


sig_classification=list(np.ones(sig[feature].size,dtype=int))
sig_classifying=pd.DataFrame(sig)
sig_classifying["Classification"]=sig_classification

bkg_classification=list(np.zeros(background[feature].size,dtype=int))
bkg_classifying=pd.DataFrame(background)
bkg_classifying["Classification"]=bkg_classification



sig_bkg=pd.concat([sig_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)



minimum_edge=min(sig_bkg[feature])
maximum_edge=max(sig_bkg[feature])
steps=100
stepsize=(maximum_edge-minimum_edge)/steps
#%%
direction='below' #use this to determine the side of the threshold that calculates TPR and FPR

threshold_direction = f'Signal is {direction} Background'
threshold=minimum_edge
TPR_arr = []
FPR_arr = []
for i in range(0,steps): 
    if threshold_direction == 'Signal is above Background':
        threshold += stepsize
       
        TP = len(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==1)][feature])#
        FP = len(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==0)][feature])
        FN = len(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==1)][feature])
        TN = len(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==0)][feature])
        
        """
        TP = sum(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        FP = sum(sig_bkg[(sig_bkg[feature] >= threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        TN = sum(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        FN = sum(sig_bkg[(sig_bkg[feature] < threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        """
        print('run',i)
        
        TPR = TP/(TP +FN)
        FPR = FP/(FP +TN)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)
        
    elif threshold_direction == 'Signal is below Background':
        threshold += stepsize
       
        TP = len(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==1)][feature])#
        FP = len(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==0)][feature])
        FN = len(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==1)][feature])
        TN = len(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==0)][feature])
        
        """
        TP = sum(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        FP = sum(sig_bkg[(sig_bkg[feature] <= threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        TN = sum(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==0)]['weight_central'])
        FN = sum(sig_bkg[(sig_bkg[feature] > threshold) & (sig_bkg['Classification']==1)]['weight_central'])
        """
        print('run',i)
        
        TPR = TP/(TP +FN)
        FPR = FP/(FP +TN)
        TPR_arr.append(TPR)
        FPR_arr.append(FPR)
        


sorted_values = sorted(zip(FPR_arr, TPR_arr))
FPR_arr, TPR_arr = zip(*sorted_values)

AUCscore=scipy.integrate.trapezoid(TPR_arr,FPR_arr,  dx=stepsize,axis=-1)
fig=plt.figure()    
plt.scatter(FPR_arr,TPR_arr,label=f'AUC:{AUCscore}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
print(AUCscore)


#%%


hist_range=(80,180) #def histogram range
#hist_range=None
feature='Diphoton_mass' #choose feature to plot




signal1='GluGluToRadionToHHTo2G2Tau_M-300' # list of one specific signal 
signal2='GluGluToRadionToHHTo2G2Tau_M-500'
signal3='GluGluToRadionToHHTo2G2Tau_M-900'

event_features=['Diphoton_mass', # list of all event features to choose from
                'Diphoton_pt_mgg','Diphoton_dPhi','LeadPhoton_pt_mgg','SubleadPhoton_pt_mgg','MET_pt','diphoton_met_dPhi','ditau_met_dPhi','ditau_deta','lead_lepton_pt','lead_lepton_mass','category','jet_1_pt','ditau_pt','ditau_mass','ditau_dR','ditau_dphi','Diphoton_ditau_dphi','dilep_leadpho_mass','event','process_id','year','MX','MY','reco_MX_mgg','Diphoton_ditau_deta','Diphoton_lead_lepton_deta','Diphoton_lead_lepton_dR','Diphoton_sublead_lepton_deta','Diphoton_sublead_lepton_dR','LeadPhoton_ditau_dR','LeadPhoton_lead_lepton_dR','SubleadPhoton_lead_lepton_dR','weight_central','weight_photon_presel_sf_Diphoton_Photon_up','weight_central_initial','weight_btag_deepjet_sf_SelectedJet_up_lfstats1','weight_btag_deepjet_sf_SelectedJet_up_cferr1','weight_muon_id_sfSYS_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_central','weight_photon_id_sf_Diphoton_Photon_up','weight_photon_presel_sf_Diphoton_Photon_down','weight_electron_veto_sf_Diphoton_Photon_up','weight_muon_iso_sfSTAT_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_down','weight_L1_prefiring_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats2','weight_muon_id_sfSTAT_SelectedMuon_up','weight_btag_deepjet_sf_SelectedJet_up_jes','weight_trigger_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats1','weight_muon_iso_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSYS_SelectedMuon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr1','weight_photon_id_sf_Diphoton_Photon_central','weight_muon_id_sfSYS_SelectedMuon_up','weight_electron_id_sf_SelectedElectron_up','weight_muon_id_sfSYS_SelectedMuon_central','weight_tau_idDeepTauVSe_sf_AnalysisTau_down','weight_puWeight_central','weight_btag_deepjet_sf_SelectedJet_down_lfstats2','weight_tau_idDeepTauVSe_sf_AnalysisTau_central','weight_tau_idDeepTauVSjet_sf_AnalysisTau_down','weight_trigger_sf_central','weight_photon_presel_sf_Diphoton_Photon_central','weight_electron_id_sf_SelectedElectron_down','weight_btag_deepjet_sf_SelectedJet_down_lf','weight_puWeight_up','weight_btag_deepjet_sf_SelectedJet_up_hfstats2','weight_btag_deepjet_sf_SelectedJet_down_lfstats1','weight_puWeight_down','weight_muon_iso_sfSYS_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_up_hfstats1','weight_tau_idDeepTauVSjet_sf_AnalysisTau_up','weight_electron_veto_sf_Diphoton_Photon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr2','weight_L1_prefiring_sf_down','weight_muon_id_sfSTAT_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_down_jes','weight_btag_deepjet_sf_SelectedJet_down_hf','weight_electron_veto_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_central','weight_btag_deepjet_sf_SelectedJet_up_hf','weight_electron_id_sf_SelectedElectron_central','weight_trigger_sf_down','weight_tau_idDeepTauVSe_sf_AnalysisTau_up','weight_tau_idDeepTauVSmu_sf_AnalysisTau_up','weight_photon_id_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_up_lf','weight_muon_id_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSTAT_SelectedMuon_up','weight_central_no_lumi','weight_btag_deepjet_sf_SelectedJet_up_cferr2','weight_btag_deepjet_sf_SelectedJet_up_lfstats2','weight_muon_iso_sfSYS_SelectedMuon_up','weight_tau_idDeepTauVSjet_sf_AnalysisTau_central','weight_L1_prefiring_sf_central']



background_list=['Data','DiPhoton', 'TTGG', 'TTGamma',#list of each bkgs for concatenation
 'TTJets',
 'VBFH_M125',
 'VH_M125',
 'WGamma',
 'ZGamma',
 'ggH_M125', 
 'ttH_M125',
 'GJets']

sig1 = df[df.process_id == proc_dict[signal1]]
sig2 = df[df.process_id == proc_dict[signal2]]
sig3 = df[df.process_id == proc_dict[signal3]]

fig=plt.figure()

plt.hist(background[feature], range=hist_range, bins=80, histtype="step", label='Concatenated Background')
plt.hist(sig1[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal1}"))
plt.hist(sig2[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal2}"))
plt.hist(sig3[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal3}"))
plt.legend(fontsize="15")
plt.show()

#%%


listforconc=[]
for i in background_list:                           #creating a concatenated list of bkg

    bkgg = df[df.process_id == proc_dict[i]]#[feature]
    listforconc.append(bkgg)
    
background = pd.concat(listforconc)


sig1_classification=list(np.ones(sig1[feature].size,dtype=int))
sig1_classifying=pd.DataFrame(sig1)
sig1_classifying["Classification"]=sig1_classification

sig2_classification=list(np.ones(sig2[feature].size,dtype=int))
sig2_classifying=pd.DataFrame(sig2)
sig2_classifying["Classification"]=sig2_classification

sig3_classification=list(np.ones(sig3[feature].size,dtype=int))
sig3_classifying=pd.DataFrame(sig3)
sig3_classifying["Classification"]=sig3_classification


bkg_classification=list(np.zeros(background[feature].size,dtype=int))
bkg_classifying=pd.DataFrame(background)
bkg_classifying["Classification"]=bkg_classification



sig_bkg1=pd.concat([sig1_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)
sig_bkg2=pd.concat([sig2_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)
sig_bkg3=pd.concat([sig3_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)





minimum_edge1=min(sig_bkg1[feature])
maximum_edge1=max(sig_bkg1[feature])
steps=100
stepsize1=(maximum_edge1-minimum_edge1)/steps

minimum_edge2=min(sig_bkg2[feature])
maximum_edge2=max(sig_bkg2[feature])
steps=100
stepsize2=(maximum_edge2-minimum_edge2)/steps

minimum_edge3=min(sig_bkg3[feature])
maximum_edge3=max(sig_bkg3[feature])
steps=100
stepsize3=(maximum_edge3-minimum_edge3)/steps
#%%
direction='above' #use this to determine the side of the threshold that calculates TPR and FPR

threshold_direction = f'Signal is {direction} Background'
threshold1=minimum_edge1
threshold2=minimum_edge2
threshold3=minimum_edge3

TPR_arr1 = []
FPR_arr1 = []
TPR_arr2 = []
FPR_arr2 = []
TPR_arr3 = []
FPR_arr3 = []

for i in range(0,steps): 
    if threshold_direction == 'Signal is above Background':
        threshold1 += stepsize1
        threshold2 += stepsize2
        threshold3 += stepsize3
        """
        TP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)][feature])#
        FP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)][feature])
        FN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)][feature])
        TN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)][feature])
        
        """
        TP1  = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
        FP1 = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
        TN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
        FN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
        
    
        
        TPR1 = TP1/(TP1 +FN1)
        FPR1 = FP1/(FP1 +TN1)
        TPR_arr1.append(TPR1)
        FPR_arr1.append(FPR1)
        """
        
        TP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)][feature])#
        FP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)][feature])
        FN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)][feature])
        TN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)][feature])
        
        """
        TP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
        FP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
        TN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
        FN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
        
        
        
        TPR2 = TP2/(TP2 +FN2)
        FPR2 = FP2/(FP2 +TN2)
        TPR_arr2.append(TPR2)
        FPR_arr2.append(FPR2)
        """
        TP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)][feature])#
        FP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)][feature])
        FN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)][feature])
        TN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)][feature])
        
        """
        TP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
        FP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
        TN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
        FN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
        
        print(i)
        
        TPR3 = TP3/(TP3 +FN3)
        FPR3 = FP3/(FP3 +TN3)
        TPR_arr3.append(TPR3)
        FPR_arr3.append(FPR3)
        
        
    elif threshold_direction == 'Signal is below Background':
        threshold1 += stepsize1
        threshold2 += stepsize2
        threshold3 += stepsize3
        """
        TP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)][feature])#
        FP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)][feature])
        FN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)][feature])
        TN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)][feature])
        
        """
        TP1  = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
        FP1 = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
        TN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
        FN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
        
        
        
        TPR1 = TP1/(TP1 +FN1)
        FPR1 = FP1/(FP1 +TN1)
        TPR_arr1.append(TPR1)
        FPR_arr1.append(FPR1)
        
        """
        TP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)][feature])#
        FP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)][feature])
        FN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)][feature])
        TN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)][feature])
        
        """
        TP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
        FP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
        TN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
        FN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
        
        
        
        TPR2 = TP2/(TP2 +FN2)
        FPR2 = FP2/(FP2 +TN2)
        TPR_arr2.append(TPR2)
        FPR_arr2.append(FPR2)
        
        """
        TP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)][feature])#
        FP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)][feature])
        FN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)][feature])
        TN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)][feature])
        
        """
        TP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
        FP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
        TN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
        FN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
        
        
        print(i)
        
        TPR3 = TP3/(TP3 +FN3)
        FPR3 = FP3/(FP3 +TN3)
        TPR_arr3.append(TPR3)
        FPR_arr3.append(FPR3)
         


sorted_values1 = sorted(zip(FPR_arr1, TPR_arr1))
FPR_arr1, TPR_arr1 = zip(*sorted_values1)


sorted_values2 = sorted(zip(FPR_arr2, TPR_arr2))
FPR_arr2, TPR_arr2 = zip(*sorted_values2)

sorted_values3 = sorted(zip(FPR_arr3, TPR_arr3))
FPR_arr3, TPR_arr3 = zip(*sorted_values3)

AUCscore1=scipy.integrate.trapezoid(TPR_arr1,FPR_arr1,  dx=stepsize,axis=-1)
AUCscore2=scipy.integrate.trapezoid(TPR_arr2,FPR_arr2,  dx=stepsize,axis=-1)
AUCscore3=scipy.integrate.trapezoid(TPR_arr3,FPR_arr3,  dx=stepsize,axis=-1)
fig=plt.figure()    
plt.scatter(FPR_arr1,TPR_arr1)
plt.scatter(FPR_arr2,TPR_arr2)
plt.scatter(FPR_arr3,TPR_arr3)
plt.plot(FPR_arr1,TPR_arr1,label=f'AUC:{AUCscore1} {signal1}')
plt.plot(FPR_arr2,TPR_arr2,label=f'AUC:{AUCscore2} {signal2}')
plt.plot(FPR_arr3,TPR_arr3,label=f'AUC:{AUCscore3} {signal3}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC Curves for {feature}')
plt.legend(fontsize='10')
plt.show()
print(AUCscore1,AUCscore2,AUCscore3)

AUCs=[feature,round(AUCscore1,4),round(AUCscore2,4),round(AUCscore3,4)]


#%%
listofAUC=[]
for i in range(0,10):
    auc=[feature,1,2,3]
    
    listofAUC.append(auc)



# Convert the list to a numpy array
array_AUC = np.array(listofAUC)

# Reshape the array into 10 rows and 3 columns
reshaped_array = array_AUC.reshape(10, 4)

print(reshaped_array)



#%%



hist_range=(80,180) #def histogram range
hist_range=None
#feature='Diphoton_mass' #choose feature to plot

listofAUCs=[]



signal1='GluGluToRadionToHHTo2G2Tau_M-300' # list of one specific signal 
signal2='GluGluToRadionToHHTo2G2Tau_M-500'
signal3='GluGluToRadionToHHTo2G2Tau_M-900'

event_features=['Diphoton_mass', # list of all event features to choose from
              'Diphoton_pt_mgg','Diphoton_dPhi','LeadPhoton_pt_mgg','SubleadPhoton_pt_mgg','MET_pt','diphoton_met_dPhi','ditau_met_dPhi','ditau_deta','lead_lepton_pt','lead_lepton_mass','category','jet_1_pt','ditau_pt','ditau_mass','ditau_dR','ditau_dphi','Diphoton_ditau_dphi','dilep_leadpho_mass','event','process_id','year','MX','MY','reco_MX_mgg','Diphoton_ditau_deta','Diphoton_lead_lepton_deta','Diphoton_lead_lepton_dR','Diphoton_sublead_lepton_deta','Diphoton_sublead_lepton_dR','LeadPhoton_ditau_dR','LeadPhoton_lead_lepton_dR','SubleadPhoton_lead_lepton_dR','weight_central','weight_photon_presel_sf_Diphoton_Photon_up','weight_central_initial','weight_btag_deepjet_sf_SelectedJet_up_lfstats1','weight_btag_deepjet_sf_SelectedJet_up_cferr1','weight_muon_id_sfSYS_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_central','weight_photon_id_sf_Diphoton_Photon_up','weight_photon_presel_sf_Diphoton_Photon_down','weight_electron_veto_sf_Diphoton_Photon_up','weight_muon_iso_sfSTAT_SelectedMuon_down','weight_tau_idDeepTauVSmu_sf_AnalysisTau_down','weight_L1_prefiring_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats2','weight_muon_id_sfSTAT_SelectedMuon_up','weight_btag_deepjet_sf_SelectedJet_up_jes','weight_trigger_sf_up','weight_btag_deepjet_sf_SelectedJet_down_hfstats1','weight_muon_iso_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSYS_SelectedMuon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr1','weight_photon_id_sf_Diphoton_Photon_central','weight_muon_id_sfSYS_SelectedMuon_up','weight_electron_id_sf_SelectedElectron_up','weight_muon_id_sfSYS_SelectedMuon_central','weight_tau_idDeepTauVSe_sf_AnalysisTau_down','weight_puWeight_central','weight_btag_deepjet_sf_SelectedJet_down_lfstats2','weight_tau_idDeepTauVSe_sf_AnalysisTau_central','weight_tau_idDeepTauVSjet_sf_AnalysisTau_down','weight_trigger_sf_central','weight_photon_presel_sf_Diphoton_Photon_central','weight_electron_id_sf_SelectedElectron_down','weight_btag_deepjet_sf_SelectedJet_down_lf','weight_puWeight_up','weight_btag_deepjet_sf_SelectedJet_up_hfstats2','weight_btag_deepjet_sf_SelectedJet_down_lfstats1','weight_puWeight_down','weight_muon_iso_sfSYS_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_up_hfstats1','weight_tau_idDeepTauVSjet_sf_AnalysisTau_up','weight_electron_veto_sf_Diphoton_Photon_central','weight_btag_deepjet_sf_SelectedJet_down_cferr2','weight_L1_prefiring_sf_down','weight_muon_id_sfSTAT_SelectedMuon_down','weight_btag_deepjet_sf_SelectedJet_down_jes','weight_btag_deepjet_sf_SelectedJet_down_hf','weight_electron_veto_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_central','weight_btag_deepjet_sf_SelectedJet_up_hf','weight_electron_id_sf_SelectedElectron_central','weight_trigger_sf_down','weight_tau_idDeepTauVSe_sf_AnalysisTau_up','weight_tau_idDeepTauVSmu_sf_AnalysisTau_up','weight_photon_id_sf_Diphoton_Photon_down','weight_btag_deepjet_sf_SelectedJet_up_lf','weight_muon_id_sfSTAT_SelectedMuon_central','weight_muon_iso_sfSTAT_SelectedMuon_up','weight_central_no_lumi','weight_btag_deepjet_sf_SelectedJet_up_cferr2','weight_btag_deepjet_sf_SelectedJet_up_lfstats2','weight_muon_iso_sfSYS_SelectedMuon_up','weight_tau_idDeepTauVSjet_sf_AnalysisTau_central','weight_L1_prefiring_sf_central']


event_features=[['Diphoton_mass',
'Diphoton_pt_mgg',
'Diphoton_dPhi',
'LeadPhoton_pt_mgg',
'SubleadPhoton_pt_mgg',
'MET_pt',
'diphoton_met_dPhi',
'lead_lepton_pt',
'lead_lepton_mass',
'jet_1_pt',
'ditau_pt',
'ditau_mass',
'dilep_leadpho_mass',
'Diphoton_lead_lepton_deta',
'Diphoton_lead_lepton_dR',
'LeadPhoton_lead_lepton_dR',
'SubleadPhoton_lead_lepton_dR',],
['above',
'above',
'above', 
'above', 
'above', 
'above', 
'above', 
'above', 
'above', 
'below',
'below',
'below',
'above', 
'above', 
'above', 
'above',
'above']]

background_list=['Data','DiPhoton', 'TTGG', 'TTGamma',#list of each bkgs for concatenation
 'TTJets',
 'VBFH_M125',
 'VH_M125',
 'WGamma',
 'ZGamma',
 'ggH_M125', 
 'ttH_M125',
 'GJets']


for feature,direction in zip(event_features[0],event_features[1]):    
    name=f'{feature}ROC'
    sig1 = df[df.process_id == proc_dict[signal1]]
    sig2 = df[df.process_id == proc_dict[signal2]]
    sig3 = df[df.process_id == proc_dict[signal3]]
    
    fig=plt.figure()
    
    plt.hist(background[feature], range=hist_range, bins=80, histtype="step", label='Concatenated Background')
    plt.hist(sig1[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal1}"))
    plt.hist(sig2[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal2}"))
    plt.hist(sig3[feature], range=hist_range, bins=80, histtype="step", label=(f"Signal {signal3}"))
    plt.legend(fontsize="15")
    plt.title(f'Histograms for {feature}')
    plt.savefig(f"C:/Users/drpla/Desktop/ICL-PHYSICS-YEAR-4/Masters Project/Data/New folder/ROCCurves/{name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    
    
# =============================================================================
#     
#     listforconc=[]
#     for i in background_list:                           #creating a concatenated list of bkg
#     
#         bkgg = df[df.process_id == proc_dict[i]]#[feature]
#         listforconc.append(bkgg)
#         
#     background = pd.concat(listforconc)
#     
#     
#     sig1_classification=list(np.ones(sig1[feature].size,dtype=int))
#     sig1_classifying=pd.DataFrame(sig1)
#     sig1_classifying["Classification"]=sig1_classification
#     
#     sig2_classification=list(np.ones(sig2[feature].size,dtype=int))
#     sig2_classifying=pd.DataFrame(sig2)
#     sig2_classifying["Classification"]=sig2_classification
#     
#     sig3_classification=list(np.ones(sig3[feature].size,dtype=int))
#     sig3_classifying=pd.DataFrame(sig3)
#     sig3_classifying["Classification"]=sig3_classification
#     
#     
#     bkg_classification=list(np.zeros(background[feature].size,dtype=int))
#     bkg_classifying=pd.DataFrame(background)
#     bkg_classifying["Classification"]=bkg_classification
#     
#     
#     
#     sig_bkg1=pd.concat([sig1_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)
#     sig_bkg2=pd.concat([sig2_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)
#     sig_bkg3=pd.concat([sig3_classifying,bkg_classifying]).sort_values(by=[feature,'Classification'], ascending=True)
#     
#     
#     
#     
#     
#     minimum_edge1=min(sig_bkg1[feature])
#     maximum_edge1=max(sig_bkg1[feature])
#     steps=100
#     stepsize1=(maximum_edge1-minimum_edge1)/steps
#     
#     minimum_edge2=min(sig_bkg2[feature])
#     maximum_edge2=max(sig_bkg2[feature])
#     steps=100
#     stepsize2=(maximum_edge2-minimum_edge2)/steps
#     
#     minimum_edge3=min(sig_bkg3[feature])
#     maximum_edge3=max(sig_bkg3[feature])
#     steps=100
#     stepsize3=(maximum_edge3-minimum_edge3)/steps
#     
#     #direction='above' #use this to determine the side of the threshold that calculates TPR and FPR
#     
#     threshold_direction = f'Signal is {direction} Background'
#     threshold1=minimum_edge1
#     threshold2=minimum_edge2
#     threshold3=minimum_edge3
#     
#     TPR_arr1 = []
#     FPR_arr1 = []
#     TPR_arr2 = []
#     FPR_arr2 = []
#     TPR_arr3 = []
#     FPR_arr3 = []
#     
#     for i in range(0,steps): 
#         if threshold_direction == 'Signal is above Background':
#             threshold1 += stepsize1
#             threshold2 += stepsize2
#             threshold3 += stepsize3
#             """
#             TP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)][feature])#
#             FP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)][feature])
#             FN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)][feature])
#             TN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)][feature])
#             
#             """
#             TP1  = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
#             FP1 = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
#             TN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
#             FN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
#             
#         
#             
#             TPR1 = TP1/(TP1 +FN1)
#             FPR1 = FP1/(FP1 +TN1)
#             TPR_arr1.append(TPR1)
#             FPR_arr1.append(FPR1)
#             """
#             
#             TP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)][feature])#
#             FP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)][feature])
#             FN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)][feature])
#             TN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)][feature])
#             
#             """
#             TP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
#             FP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
#             TN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
#             FN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
#             
#             
#             
#             TPR2 = TP2/(TP2 +FN2)
#             FPR2 = FP2/(FP2 +TN2)
#             TPR_arr2.append(TPR2)
#             FPR_arr2.append(FPR2)
#             """
#             TP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)][feature])#
#             FP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)][feature])
#             FN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)][feature])
#             TN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)][feature])
#             
#             """
#             TP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
#             FP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
#             TN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
#             FN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
#             
#             print(i)
#             
#             TPR3 = TP3/(TP3 +FN3)
#             FPR3 = FP3/(FP3 +TN3)
#             TPR_arr3.append(TPR3)
#             FPR_arr3.append(FPR3)
#             
#             
#         elif threshold_direction == 'Signal is below Background':
#             threshold1 += stepsize1
#             threshold2 += stepsize2
#             threshold3 += stepsize3
#             """
#             TP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)][feature])#
#             FP1 = len(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)][feature])
#             FN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)][feature])
#             TN1 = len(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)][feature])
#             
#             """
#             TP1  = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
#             FP1 = sum(sig_bkg1[(sig_bkg1[feature] >= threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
#             TN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==0)]['weight_central'])
#             FN1 = sum(sig_bkg1[(sig_bkg1[feature] < threshold1) & (sig_bkg1['Classification']==1)]['weight_central'])
#             
#             
#             
#             TPR1 = TP1/(TP1 +FN1)
#             FPR1 = FP1/(FP1 +TN1)
#             TPR_arr1.append(TPR1)
#             FPR_arr1.append(FPR1)
#             
#             """
#             TP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)][feature])#
#             FP2 = len(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)][feature])
#             FN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)][feature])
#             TN2 = len(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)][feature])
#             
#             """
#             TP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
#             FP2 = sum(sig_bkg2[(sig_bkg2[feature] >= threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
#             TN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==0)]['weight_central'])
#             FN2 = sum(sig_bkg2[(sig_bkg2[feature] < threshold2) & (sig_bkg2['Classification']==1)]['weight_central'])
#             
#             
#             
#             TPR2 = TP2/(TP2 +FN2)
#             FPR2 = FP2/(FP2 +TN2)
#             TPR_arr2.append(TPR2)
#             FPR_arr2.append(FPR2)
#             
#             """
#             TP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)][feature])#
#             FP3 = len(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)][feature])
#             FN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)][feature])
#             TN3 = len(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)][feature])
#             
#             """
#             TP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
#             FP3 = sum(sig_bkg3[(sig_bkg3[feature] >= threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
#             TN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==0)]['weight_central'])
#             FN3 = sum(sig_bkg3[(sig_bkg3[feature] < threshold3) & (sig_bkg3['Classification']==1)]['weight_central'])
#             
#             
#             print(i)
#             
#             TPR3 = TP3/(TP3 +FN3)
#             FPR3 = FP3/(FP3 +TN3)
#             TPR_arr3.append(TPR3)
#             FPR_arr3.append(FPR3)
#              
#     
#     
#     sorted_values1 = sorted(zip(FPR_arr1, TPR_arr1))
#     FPR_arr1, TPR_arr1 = zip(*sorted_values1)
#     
#     
#     sorted_values2 = sorted(zip(FPR_arr2, TPR_arr2))
#     FPR_arr2, TPR_arr2 = zip(*sorted_values2)
#     
#     sorted_values3 = sorted(zip(FPR_arr3, TPR_arr3))
#     FPR_arr3, TPR_arr3 = zip(*sorted_values3)
#     
#     AUCscore1=scipy.integrate.trapezoid(TPR_arr1,FPR_arr1,  dx=stepsize,axis=-1)
#     AUCscore2=scipy.integrate.trapezoid(TPR_arr2,FPR_arr2,  dx=stepsize,axis=-1)
#     AUCscore3=scipy.integrate.trapezoid(TPR_arr3,FPR_arr3,  dx=stepsize,axis=-1)
#     fig=plt.figure()    
#     plt.scatter(FPR_arr1,TPR_arr1)
#     plt.scatter(FPR_arr2,TPR_arr2)
#     plt.scatter(FPR_arr3,TPR_arr3)
#     plt.plot(FPR_arr1,TPR_arr1,label=f'AUC:{AUCscore1} {signal1}')
#     plt.plot(FPR_arr2,TPR_arr2,label=f'AUC:{AUCscore2} {signal2}')
#     plt.plot(FPR_arr3,TPR_arr3,label=f'AUC:{AUCscore3} {signal3}')
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.title(f'ROC Curves for {feature}')
#     plt.legend(fontsize='10')
#     plt.savefig(f"C:/Users/drpla/Desktop/ICL-PHYSICS-YEAR-4/Masters Project/Data/New folder/ROCCurves/{name}.pdf", format="pdf", bbox_inches="tight")
#     plt.show()
#     print(feature, AUCscore1,AUCscore2,AUCscore3)
#     
#     AUCs=[feature,round(AUCscore1,4),round(AUCscore2,4),round(AUCscore3,4)]
#     listofAUCs.append(AUCs)
# =============================================================================

