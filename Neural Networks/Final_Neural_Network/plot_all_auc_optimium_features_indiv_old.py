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



num = 18
optimum_features_test = pd.read_csv(f"optimium_features_v2_{num}.csv")
signals_masses = list(optimum_features_test.keys())
auc_test = []


for key in signals_masses:
    auc_test.append(optimum_features_test[key].item())
    


plt.scatter(signals_masses,auc_test, marker= 'o',label=f'Test {num} Features')
plt.ylabel('AUC Score',fontsize=10)
plt.xlabel('Signal Mass (GeV)',fontsize=10)
plt.title('Plot of optimal features against their AUC scores',fontsize=15)
plt.legend(fontsize=10)
plt.tight_layout()
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.grid()
plt.savefig(f'Neural Networks\Final_Neural_Network\Plots\optimium_features_high_lr_f_{num}.png', format='png')
plt.show()



