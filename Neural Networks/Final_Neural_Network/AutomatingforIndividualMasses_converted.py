#################### New Cell ####################
"""
Generating neural networks for individual masses 
Automated across each feature to allow direct comparison

allows investigation into learning rate and architecture
"""

#################### New Cell ####################
import torch

#%matplotlib nbagg
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import mplhep as hep
from sklearn.metrics import roc_curve, auc

hep.style.use("CMS")

df = pd.read_parquet(r'C:\Users\drpla\Desktop\ICL-PHYSICS-YEAR-4\Masters Project\Data\New folder\merged_nominal.parquet')


with open(r'C:\Users\drpla\Desktop\ICL-PHYSICS-YEAR-4\Masters Project\Data\New folder\summary.json', "r") as f:
  proc_dict = json.load(f)["sample_id_map"]
  

#################### New Cell ####################
featurelist=['reco_MX_mgg','Diphoton_pt_mgg','LeadPhoton_pt_mgg','ditau_pt','Diphoton_dPhi','dilep_leadpho_mass','lead_lepton_pt','MET_pt','ditau_dR','SubleadPhoton_pt_mgg','Diphoton_lead_lepton_deta','ditau_met_dPhi','ditau_deta','Diphoton_sublead_lepton_deta','Diphoton_ditau_deta','ditau_mass']
#featurelist=['reco_MX_mgg','Diphoton_pt_mgg']
featurelist=['Diphoton_mass', 'Diphoton_pt_mgg', 'Diphoton_dPhi',
       'LeadPhoton_pt_mgg', 'SubleadPhoton_pt_mgg', 'MET_pt',
       'diphoton_met_dPhi', 'ditau_met_dPhi', 'ditau_deta', 'lead_lepton_pt',
       'lead_lepton_mass', 'jet_1_pt', 'ditau_pt', 'ditau_mass',
       'ditau_dR', 'ditau_dphi', 'Diphoton_ditau_dphi', 'dilep_leadpho_mass','reco_MX_mgg',
       'Diphoton_ditau_deta', 'Diphoton_lead_lepton_deta',
       'Diphoton_lead_lepton_dR', 'Diphoton_sublead_lepton_deta',
       'Diphoton_sublead_lepton_dR', 'LeadPhoton_ditau_dR',
       'LeadPhoton_lead_lepton_dR', 'SubleadPhoton_lead_lepton_dR']

savefig='not yes'
writeAUCtocsv='not yes'
epochs=200
lr=0.03
learningrate='0_03'
aucscoreslist=[]

signalname='GluGluToRadionToHHTo2G2Tau_M-1000'
mass='Mx_1000'
name=0


#################### New Cell ####################
featurelist

#################### New Cell ####################
#for thing in signalnames:    
for featurename in featurelist: 
    #signalname='GluGluToRadionToHHTo2G2Tau_M-700'
    sig = df[df.process_id == proc_dict[f"{signalname}"]] # just one signal process, mass of X is 1000 GeV
    sig['Classification']=np.ones(sig['Diphoton_mass'].size)
    """Concatenating the background data"""
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
        bkgg = df[df.process_id == proc_dict[i]]
        listforconc.append(bkgg)

    background = pd.concat(listforconc)
    background['Classification']=np.zeros(background['Diphoton_mass'].size)

    """The features requiring exclusion of -9 values"""
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

    """Concatenating Signal and Background"""
    """Choosing Best Features given the M=1000 AUC scores"""
    FullSignalBackground=pd.concat([sig,background])

    df_TopFeatures=pd.DataFrame()

    TopFeatures=featurename
    """A dataset consisting of only the essential features"""
    #for feature in TopFeatures:
    df_TopFeatures[featurename]=FullSignalBackground[featurename]
    df_TopFeatures['Classification']=FullSignalBackground['Classification']
    df_TopFeatures['weight_central']=FullSignalBackground['weight_central']
     
    """Removal of the values that are binned at -9 from the necessary features"""
    for columns in df_TopFeatures.columns:
        if columns in MinusNineBinning:
            df_TopFeatures = df_TopFeatures.loc[(df_TopFeatures[columns] > -8)]

    df_TopFeatures = df_TopFeatures.sample(frac=1, random_state=42)  # Setting frac=1 shuffles all rows

    features = df_TopFeatures # Extracting features

    labels = df_TopFeatures['Classification']  # Extracting labels

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=(1/3), random_state=42)
    test_weights=pd.DataFrame()
    train_weights=pd.DataFrame()

    weightofsignal=train_features[train_features['Classification']==1]['weight_central'].sum()
    weightofbackground=train_features[train_features['Classification']==0]['weight_central'].sum()
    scale=weightofsignal/weightofbackground

    """reweighting the weight_central column in entire data 
    set such that for background and signal """
    train_features.loc[train_features['Classification'] == 0, 'weight_central'] *= scale
    test_features.loc[test_features['Classification'] == 0, 'weight_central'] *= scale


    train_weights['weight_central']=train_features['weight_central']
    test_weights['weight_central']=test_features['weight_central']



    train_features = train_features.drop(columns=['weight_central'])
    train_features = train_features.drop(columns=['Classification'])
    test_features=test_features.drop(columns=['weight_central'])

    train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
    train_weights_tensor = torch.tensor(train_weights.values,dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels.values,dtype=torch.float32)

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN,self).__init__()
            self.hidden1 = nn.Linear(1, 8)
            self.act1 = nn.ReLU()
    #         self.hidden2 = nn.Linear(20, 8)
    #         self.act2 = nn.ReLU()
    #         self.output = nn.Linear(8, 1)
    #         self.hidden1 = nn.Linear(16, 8)
    #         self.act1 = nn.ReLU()
    #         self.hidden2 = nn.Linear(20, 40)
    #         self.act2 = nn.ReLU()
    #         self.hidden3 = nn.Linear(40, 16)
    #         self.act3 = nn.ReLU()
    #         self.hidden4 = nn.Linear(16, 8)
    #         self.act4 = nn.ReLU()
            self.output = nn.Linear(8, 1)
            self.act_output = nn.Sigmoid()

        def forward(self, x):
            x = self.hidden1(x)
            x = self.act1(x)
          #  x = self.hidden2(x)
          #  x = self.act2(x)
          #  x = self.hidden3(x)
          #  x = self.act3(x)
          #  x = self.hidden4(x)
          #  x = self.act4(x)
            x = self.output(x)
            x = self.act_output(x)
            return x


        def weightedBCELoss(self, input, target, weight):
          x, y, w = input, target, weight
          log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
          #return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))
          return -w * (y*log(x) + (1-y)*log(1-x))

        def batch_weightedBCELoss(self, input, target, weight, batch_size):
    #batch_weightedBCELoss(self, train, train_labels_tensor, train_weights_tensor, batch_size)
            self.batch_size=batch_size

            target=target.unsqueeze(1)


           # train=model.forward(input)

            total_batch_err=torch.empty(0,1)
            output_length=input.shape[0]
            batch_remainder=output_length%batch_size

            for i in range(0, output_length//batch_size):
                weights = weight[i*(batch_size):(i+1)*(batch_size), :]
                labels = target[i*(batch_size):(i+1)*(batch_size), :]
                inputs = input[i*(batch_size):(i+1)*(batch_size), :]

                loss=self.weightedBCELoss(inputs, labels, weights)

                total_batch_err=torch.cat((total_batch_err,loss)) 
            #    print(total_batch_err.shape[0])

            if batch_remainder > 0:
                weights = weight[(output_length//batch_size)*batch_size:, :]
                labels = target[(output_length//batch_size)*batch_size:, :]
                inputs = input[(output_length//batch_size)*batch_size:, :]

                loss=self.weightedBCELoss(inputs, labels, weights)

                #weights = train_weights_tensor[(train_weights_tensor.shape[0]//batch_size)*batch_size:, :]
                total_batch_err=torch.cat((total_batch_err,loss))
            #    print(total_batch_err.shape[0])

            return torch.mean(total_batch_err)


    model = SimpleNN()
    #lr=0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    lossdata=[]
    df_Prediction=pd.DataFrame()
    #    epochs=200
    epochlist=[]
    for i in range(1,epochs+1):
        epochlist.append(i)
#     for i in range(0,epochs):
#         trained=model.forward(train_features_tensor)
#         trained_data= pd.DataFrame(trained.detach().numpy())
#         df_Prediction[f'Epoch {i}'] = trained_data.copy()
#         loss=model.batch_weightedBCELoss(trained,train_labels_tensor,train_weights_tensor,1024)
#         lossdata.append(loss.item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # print(f'For Epoch {i+1}: Loss = {loss}')
    trained_data_list = []
    for i in range(0,epochs):
        trained=model.forward(train_features_tensor)
        trained_data= pd.DataFrame(trained.detach().numpy())
        trained_data_list.append(trained_data.copy())  # Append trained_data to the list

    #    df_Prediction[f'Epoch {i}'] = trained_data
        loss=model.batch_weightedBCELoss(trained,train_labels_tensor,train_weights_tensor,1024)
        lossdata.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'For Epoch {i+1}: Loss = {loss}')
    df_Prediction = pd.concat(trained_data_list, axis=1)
    df_Prediction.columns = [f'Epoch {i}' for i in range(epochs)]



    # figure=plt.figure()
    # plt.plot(epochlist,lossdata)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()    


    """
    Plot of Loss vs Epoch
    """

    # figure=plt.figure()
    # plt.plot(epochlist,lossdata)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()    

    #epoch_= 'Epoch 99'
    """Adding the labels to the prediction dataframe"""
    #epoch_= 'Epoch 99'
    epoch_=f'Epoch {epochs-1}'
    train_labels_=pd.DataFrame({'Classification': train_labels}).reset_index(drop=True)
    df_Prediction = pd.concat([df_Prediction, train_labels_], axis=1)
    epoch_=f'Epoch {epochs-1}'
    # plt.figure()
    # plt.hist(df_Prediction[df_Prediction['Classification']==1][epoch_],bins=80,label='predicted signal',histtype="step")
    # plt.hist(df_Prediction[df_Prediction['Classification']==0][epoch_],bins=80,histtype='step',label='predicted background')
    # plt.legend()
    # plt.xlabel('Classification of Events')
    # plt.ylabel('Number of Events')
    # plt.title('Comparison of the expected output and the trained output')
    # #plt.savefig(f"BenNeuralNetworkPlots/TrainingHist-{signalname}Epochs={epochs}")
    # plt.show()

    df_Prediction.sort_values(by=[epoch_,'Classification'], ascending=True)
    fpr, tpr, thresholds = roc_curve(df_Prediction['Classification'], df_Prediction[epoch_])
    roc_auc = auc(fpr, tpr)
    fig=plt.figure()    
    plt.plot(fpr,tpr,label=f'AUC:{roc_auc}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
    plt.legend()
    plt.show()


    # #epoch_= 'Epoch 99'
    # epoch_=f'Epoch {epochs-1}'
    # plt.figure()
    # plt.hist(df_Prediction[df_Prediction['Classification']==1][epoch_],bins=80,label='predicted signal',histtype="step")
    # plt.hist(df_Prediction[df_Prediction['Classification']==0][epoch_],bins=80,histtype='step',label='predicted background')
    # plt.legend()
    # plt.xlabel('Classification of Events')
    # plt.ylabel('Number of Events')
    # plt.title('Comparison of the expected output and the trained output')
    # #plt.savefig(f"BenNeuralNetworkPlots/TrainingHist-{signalname}Epochs={epochs}")
    # plt.show()



    fig, axs = plt.subplots(1, 3, figsize=(24, 10))

    # Plot 1: Line plot (plt.plot)
    axs[1].plot(epochlist, lossdata)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title(f'Loss per Epoch: {TopFeatures}',fontsize=18)

    # Plot 2: Histograms (two separate subplots

    axs[0].set_title('Trained Histogram',fontsize=18)

    axs[0].set_xlabel('Trained Classification')
    axs[0].set_ylabel('Number Events')

    axs[0].hist(df_Prediction[df_Prediction['Classification']==1][epoch_],bins=80,label='predicted signal',histtype="step",range=(0,1))
    axs[0].hist(df_Prediction[df_Prediction['Classification']==0][epoch_],bins=80,histtype='step',label='predicted background',range=(0,1))
    axs[0].legend()
    axs[0].set_title('Expected Histogram',fontsize=18)

    # Plot 3: Scatter plot with AUC score
    # axs[2].scatter(FPR_arr, TPR_arr, label=f'AUC: {AUCscore}')
    # axs[2].set_xlabel('FPR')
    # axs[2].set_ylabel('TPR')
    # axs[2].legend()
    # axs[2].set_title('ROC Curve')

    axs[2].plot(fpr,tpr,label=f'AUC:{roc_auc}')
    axs[2].set_xlabel('FPR')
    axs[2].set_ylabel('TPR')
    axs[2].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guessing')
    axs[2].legend()


    # Adjust layout and display
    plt.tight_layout()
    #plt.savefig(f"BenNeuralNetworkPlots/reco_MX_mgg_SimArch/HIST_LOSS_ROC_{signalname}=_Feature={TopFeatures[0]}_Epochs={epochs}_1")
    if savefig == 'yes':
        plt.savefig(f"Figures{TopFeatures}/HIST_LOSS_ROC_{signalname}=_Feat={TopFeatures}_Epo={epochs}_lr={learningrate}")
    print(f"Figures{TopFeatures}/HIST_LOSS_ROC_{signalname}=_Feat={TopFeatures}_Epo={epochs}_lr={learningrate}")

    plt.show()
    print(roc_auc)
    aucscoreslist.append(roc_auc)
    name+=1
df_AucScores = pd.DataFrame({'Column': aucscoreslist})
# File path to write CSV
file_path = f'AucScores_{TopFeatures}.csv'
if writeAUCtocsv=='yes':
    df_AucScores.to_csv(file_path, index=False)


#################### New Cell ####################
#masses=[1000,900,800,750,700,650,600,550,500,450,400,350,320,300,290,280,270,260]

plt.plot(featurelist[0:name],aucscoreslist,'o')
plt.xlabel('Features',fontsize=12)
plt.ylabel('AUC Score',fontsize=12)
plt.xticks(rotation=90,fontsize=12)
plt.title(f'{signalname}: Plot of AUC Score for each feature at {epochs} epochs',fontsize=12)

savefig='no'
if savefig == 'yes':
    plt.savefig(f"C:\\Users\\drpla\\Desktop\\ICL-PHYSICS-YEAR-4\\Masters Project\\GIT\\CMS_Physics_PNN\\Neural Networks\\Ben Neural Network\\NNOffline\\AUC_v_Feature\\{mass}\\AUC_v_Feature_Mx={signalname}_epochs={epochs}_lr={learningrate}")
plt.show()   

#################### New Cell ####################
feature_list = ['Diphoton_mass', 'Diphoton_pt_mgg', 'Diphoton_dPhi',
       'LeadPhoton_pt_mgg', 'SubleadPhoton_pt_mgg', 'MET_pt',
       'diphoton_met_dPhi', 'ditau_met_dPhi', 'ditau_deta', 'lead_lepton_pt',
       'lead_lepton_mass', 'jet_1_pt', 'ditau_pt', 'ditau_mass',
       'ditau_dR', 'ditau_dphi', 'Diphoton_ditau_dphi', 'dilep_leadpho_mass','reco_MX_mgg',
       'Diphoton_ditau_deta', 'Diphoton_lead_lepton_deta',
       'Diphoton_lead_lepton_dR', 'Diphoton_sublead_lepton_deta',
       'Diphoton_sublead_lepton_dR', 'LeadPhoton_ditau_dR',
       'LeadPhoton_lead_lepton_dR', 'SubleadPhoton_lead_lepton_dR']
#featurelist=['reco_MX_mgg','Diphoton_pt_mgg','LeadPhoton_pt_mgg','ditau_pt','Diphoton_dPhi','dilep_leadpho_mass','lead_lepton_pt','MET_pt','ditau_dR','SubleadPhoton_pt_mgg','Diphoton_lead_lepton_deta','ditau_met_dPhi','ditau_deta','Diphoton_sublead_lepton_deta','Diphoton_ditau_deta','ditau_mass']

#GluGluToRadionToHHTo2G2Tau_M_1000_AUC = pd.read_csv("GluGluToRadionToHHTo2G2Tau_M_1000_AUC.csv")
GluGluToRadionToHHTo2G2Tau_M_1000_AUC = pd.read_csv(f'C:\\Users\\drpla\\Desktop\\ICL-PHYSICS-YEAR-4\\Masters Project\\GIT\\CMS_Physics_PNN\\Threshold Analysis\\Ganel Threshold Analysis\\GluGluToRadionToHHTo2G2Tau_M_1000_AUC.csv')

GluGluToRadionToHHTo2G2Tau_M_1000_AUC = GluGluToRadionToHHTo2G2Tau_M_1000_AUC[featurelist]


GluGluToRadionToHHTo2G2Tau_M_1000_AUC_np = (GluGluToRadionToHHTo2G2Tau_M_1000_AUC.iloc[0]).to_numpy()

aucscoreslist_np = np.array(aucscoreslist)

# temp_dict = {'Features': feature_list, 'GluGluToRadionToHHTo2G2Tau_M_300':GluGluToRadionToHHTo2G2Tau_M_300_AUC_np,
#              'GluGluToRadionToHHTo2G2Tau_M_400':GluGluToRadionToHHTo2G2Tau_M_400_AUC_np, 'GluGluToRadionToHHTo2G2Tau_M_600':GluGluToRadionToHHTo2G2Tau_M_600_AUC_np,
#              'GluGluToRadionToHHTo2G2Tau_M_900':GluGluToRadionToHHTo2G2Tau_M_900_AUC_np,'GluGluToRadionToHHTo2G2Tau_M_800':GluGluToRadionToHHTo2G2Tau_M_800_AUC_np,
#              'GluGluToRadionToHHTo2G2Tau_M_1000':GluGluToRadionToHHTo2G2Tau_M_1000_AUC_np,'NN_GluGluToRadionToHHTo2G2Tau_M_1000': NN_GluGluToRadionToHHTo2G2Tau_M_1000_AUC_np}
temp_dict = {'Features': feature_list, 'GluGluToRadionToHHTo2G2Tau_M_1000':GluGluToRadionToHHTo2G2Tau_M_1000_AUC_np,'NN_GluGluToRadionToHHTo2G2Tau_M_1000': aucscoreslist_np}

temp_dict_df = pd.DataFrame(data=temp_dict)

reorder_dict_df=temp_dict_df.sort_values(by=['GluGluToRadionToHHTo2G2Tau_M_1000'], ascending=False)
i=0

for columns in reorder_dict_df.columns:
    if i>0:
        plt.plot(reorder_dict_df['Features'],reorder_dict_df[columns])
        plt.scatter(reorder_dict_df['Features'],reorder_dict_df[columns],label=f'{columns}')
        plt.ylabel('AUC Score',fontsize=10)
        plt.xlabel('Event Features',fontsize=10)
        plt.title('Plot of features against their AUC scores (sorted by M=1000 GeV)',fontsize=15)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.xticks(rotation=90,fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid()
    i+=1

#################### New Cell ####################
TopFeatures

#################### New Cell ####################


#################### New Cell ####################


