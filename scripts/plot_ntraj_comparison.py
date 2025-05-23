# SPDX-FileCopyrightText: 2025 University of Manchester
#
# SPDX-License-Identifier: apache-2.0

"""
Script to go through various folders of model runs
and compare different ntraj evaluations.
"""

import pandas as pd
import matplotlib.pyplot as plt

# %%
eps_list = ["nodp","eps1","eps3","eps5","eps10"]
labels = ["non-DP","eps1","eps3","eps5","eps10"]
colors = ['k', 'firebrick', 'orange', 'limegreen', 'blue']

# %%

fig,ax = plt.subplots()

num_train_users = 50
for eps,color,label in zip(eps_list,colors,labels):
        # Load CSV
        filepath_eval_ntraj = f'models/ex2/{num_train_users}/{eps}/eval_acc_vs_ntraj.csv'
        
        df_eval_ntraj = pd.read_csv(filepath_eval_ntraj)
        df_eval_ntraj.set_index('n_traj',inplace=True)
        
        ax.plot(df_eval_ntraj.index,df_eval_ntraj['acc_sample_mean'], marker='o', color=color, linestyle='--', label=label)
        ax.fill_between(df_eval_ntraj.index, 
                        df_eval_ntraj['acc_sample_Q25'], 
                        df_eval_ntraj['acc_sample_Q75'], 
                        color=color, 
                        alpha=0.1)

ax.legend(fontsize=16)

# Set the labels and title, legend etc.
plt.xlabel('Behaviours seen',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)
plt.grid(True)
plt.tight_layout()  

# Show the plot
plt.savefig('figures/ex2/ex2_eval_ntraj.png',dpi=200)
plt.show()

