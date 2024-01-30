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




optimum_features_10 = pd.read_csv(f"optimium_features_v2_10_lr_1e-05_train.csv")
optimum_features_12 = pd.read_csv(f"optimium_features_v2_12_lr_1e-05_train.csv")
optimum_features_14 = pd.read_csv(f"optimium_features_v2_14_lr_1e-05_train.csv")
optimum_features_16 = pd.read_csv(f"optimium_features_v2_16_lr_1e-05_train.csv")
optimum_features_18 = pd.read_csv(f"optimium_features_v2_18_lr_1e-05_train.csv")
signals_masses = list(optimum_features_10.keys())
auc_10 = []
auc_12 = []
auc_14 = []
auc_16 = []
auc_18 = []
for key in signals_masses:
    auc_10.append(optimum_features_10[key].item())
    auc_12.append(optimum_features_12[key].item())
    auc_14.append(optimum_features_14[key].item())
    auc_16.append(optimum_features_16[key].item())
    auc_18.append(optimum_features_18[key].item())
    


plt.plot(signals_masses,auc_10, marker= 'o',label=f'10 Features')
plt.plot(signals_masses,auc_12, marker= 'o',label=f'12 Features')
plt.plot(signals_masses,auc_14, marker= 'o',label=f'14 Features')
plt.plot(signals_masses,auc_16, marker= 'o',label=f'16 Features')
plt.plot(signals_masses,auc_18, marker= 'o',label=f'18 Features')
plt.ylabel('AUC Score',fontsize=10)
plt.xlabel('Signal Mass (GeV)',fontsize=10)
plt.title('Plot of optimal features against their AUC scores',fontsize=15)
plt.legend(fontsize=10)
plt.tight_layout()
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.show()



