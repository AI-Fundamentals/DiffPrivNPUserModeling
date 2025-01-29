"""
Script to go through various folders of model runs,
work out the best epoch (i.e. highest accuracy),
and save that in settings files for eval_ntraj.
"""

import pandas as pd
import matplotlib.pyplot as plt

# %%
eps_list = ["eps1","eps3","eps5","eps10","nodp"]
num_train_users_list = [150,300,600,900,1200,1500,1800]

# %%

df_best_epoch = pd.DataFrame(index=num_train_users_list,columns=eps_list)
df_best_acc = pd.DataFrame(index=num_train_users_list,columns=eps_list)

for eps in eps_list:
    for num_train_users in num_train_users_list:
        try:
            # Load CSV
            filepath = f'models/ex2/{num_train_users}/{eps}/eval_acc_vs_epochs.csv'
            df_acc = pd.read_csv(filepath)
            df_acc.set_index('epoch',inplace=True)
        
            # Find the best epoch
            best_epoch = df_acc['acc_sample_mean'].argmax()
            best_acc = df_acc['acc_sample_mean'][best_epoch]            
            
            df_best_epoch.loc[num_train_users][eps] = best_epoch
            df_best_acc.loc[num_train_users][eps] = best_acc
        except:
            pass

# %% Make the plot

# Plot the data
colors = ['firebrick','grey','limegreen','blue','k']
df_best_acc.plot(kind='line', marker='o',color=colors)

# Set the labels and title
plt.xlabel('n_users')
plt.ylabel('Accuracy')
plt.title('Best epoch, 1000 validation users')

# Show the plot
plt.savefig('figures/ex2/val_acc_best_epoch.png',dpi=300)
plt.show()
















# # %%
# # Load base settings file
# base_settings_path = f'settings/ex2/{num_train_users}/settings_ex2_eval_ntraj.json'

# # Open a JSON file and load its contents into a Python dictionary
# with open(base_settings_path, 'r') as file:
#     settings = json.load(file)

# settings['num_users'] = 5000
# settings['batch_size'] = 1
# settings['models_dir'] = f'models/ex2/ex2/{num_train_users}/{eps}/'
# settings['figs_dir'] = f'figures/ex2/{num_train_users}/{eps}/'
# settings['init_weights'] = f'models/ex2/ex2/{num_train_users}/{eps}/weights_epoch_{best_epoch}.pt'

# # %%

# output_folder = f'settings/ex2/{num_train_users}/'
# output_filename = f'settings_ex2_eval_ntraj_{eps}_TEST.json'

# with open(output_folder+output_filename, 'w') as file:
#     json.dump(settings, file, indent=4)
