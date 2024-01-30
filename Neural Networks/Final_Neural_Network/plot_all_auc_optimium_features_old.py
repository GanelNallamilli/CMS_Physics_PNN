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




optimum_features_3 = pd.read_csv(f"optimium_features_3.csv")
optimum_features_5 = pd.read_csv(f"optimium_features_5.csv")
optimum_features_7 = pd.read_csv(f"optimium_features_7.csv")
optimum_features_9 = pd.read_csv(f"optimium_features_9.csv")
optimum_features_12 = pd.read_csv(f"optimium_features_12.csv")
optimum_features_15 = pd.read_csv(f"optimium_features_15.csv")
optimum_features_20 = pd.read_csv(f"optimium_features_20.csv")
signals_masses = list(optimum_features_3.keys())
auc_3 = []
auc_5 = []
auc_7 = []
auc_9 = []
auc_10 = []
auc_12 = []
auc_15 = []
auc_20 = []

for key in signals_masses:
    auc_3.append(optimum_features_3[key].item())
    auc_5.append(optimum_features_5[key].item())
    auc_7.append(optimum_features_7[key].item())
    auc_9.append(optimum_features_9[key].item())
    auc_12.append(optimum_features_12[key].item())
    auc_15.append(optimum_features_15[key].item())
    auc_20.append(optimum_features_20[key].item())
    


plt.plot(signals_masses,auc_3, marker= 'o',label=f'3 Features')
plt.plot(signals_masses,auc_5, marker= 'o',label=f'5 Features')
plt.plot(signals_masses,auc_7, marker= 'o',label=f'7 Features')
plt.plot(signals_masses,auc_9, marker= 'o',label=f'9 Features')
plt.plot(signals_masses,auc_12, marker= 'o',label=f'12 Features')
plt.plot(signals_masses,auc_15, marker= 'o',label=f'15 Features')
plt.plot(signals_masses,auc_20, marker= 'o',label=f'20 Features')
plt.ylabel('AUC Score',fontsize=10)
plt.ylim(bottom = 0.75)
plt.xlabel('Signal Mass (GeV)',fontsize=10)
plt.title('Plot of optimal features against their AUC scores',fontsize=15)
plt.legend(fontsize=10)
plt.tight_layout()
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.savefig(f'Neural Networks\Final_Neural_Network\Plots\optimium_features_all_old.png', format='png')
plt.show()



