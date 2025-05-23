# SPDX-FileCopyrightText: 2025 University of Manchester
#
# SPDX-License-Identifier: apache-2.0

"""
Script to go through various folders of model runs,
work out the best epoch (i.e. highest accuracy),
and plot for the different values of epsilon.
"""

import pandas as pd
import matplotlib.pyplot as plt

# %%
eps_list = ["nodp", "eps1", "eps3", "eps5", "eps10"]
labels = ["non-DP", "eps1", "eps3", "eps5", "eps10"]

# Change this value to change the experiment
experiment = 3


if experiment == 2:
    num_train_users_list = [50, 100, 150, 300, 600, 900, 1200, 1500, 1800]
elif experiment == 3:
    num_train_users_list = [300]

# %%

df_best_epoch = pd.DataFrame(index=num_train_users_list, columns=eps_list)
df_best_acc = pd.DataFrame(index=num_train_users_list, columns=eps_list)
df_best_acc_Q25 = pd.DataFrame(index=num_train_users_list, columns=eps_list)
df_best_acc_Q75 = pd.DataFrame(index=num_train_users_list, columns=eps_list)

for eps in eps_list:
    for num_train_users in num_train_users_list:
        try:
            # Load CSV
            filepath_train = f'models/ex{experiment}/{num_train_users}/{eps}/training_metrics.csv'
            filepath_val = f'models/ex{experiment}/{num_train_users}/{eps}/val_acc_vs_epochs.csv'
            df_train = pd.read_csv(filepath_train)
            df_train.set_index('epoch', inplace=True)
            df_val = pd.read_csv(filepath_val)
            df_val.set_index('epoch', inplace=True)

            # Find the best epoch
            # Alternative method left in but commented out
            # best_epoch = df_val['acc_sample_mean'].argmax()
            best_epoch = df_train['loss'].argmin()
            best_acc = df_val['acc_sample_mean'][best_epoch]
            best_acc_Q25 = df_val['acc_sample_Q25'][best_epoch]
            best_acc_Q75 = df_val['acc_sample_Q75'][best_epoch]

            df_best_epoch.loc[num_train_users][eps] = best_epoch
            df_best_acc.loc[num_train_users][eps] = best_acc
            df_best_acc_Q25.loc[num_train_users][eps] = best_acc_Q25
            df_best_acc_Q75.loc[num_train_users][eps] = best_acc_Q75

        except:
            pass


df_best_epoch.to_csv(f'models/ex{experiment}/ex{experiment}_best_epochs.csv')
df_best_acc.to_csv(f'models/ex{experiment}/ex{experiment}_best_acc_mean.csv')
df_best_acc_Q25.to_csv(
    f'models/ex{experiment}/ex{experiment}_best_acc_Q25.csv')
df_best_acc_Q75.to_csv(
    f'models/ex{experiment}/ex{experiment}_best_acc_Q75.csv')


# %% Convert the dataframes to numeric type to avoid type issues
df_best_acc = df_best_acc.apply(pd.to_numeric)
df_best_acc_Q25 = df_best_acc_Q25.apply(pd.to_numeric)
df_best_acc_Q75 = df_best_acc_Q75.apply(pd.to_numeric)

# %% Make the plot
colors = ['k', 'firebrick', 'orange', 'limegreen', 'blue']

# Plot the mean
ax = df_best_acc.plot(kind='line', marker='o', color=colors, linestyle='--')

# Plot the error bars
for eps, color in zip(eps_list, colors):
    ax.fill_between(num_train_users_list,
                    df_best_acc_Q25[eps],
                    df_best_acc_Q75[eps],
                    color=color,
                    alpha=0.1)

# Set the labels and title, legend etc.
plt.legend(labels, fontsize=16)
plt.xlabel('Users seen', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig(
    f'figures/ex{experiment}/ex{experiment}_val_acc_best_epoch.png', dpi=200)
plt.show()
