# CSF Jobscripts

This folder contains jobscripts designed to run the code on the University of Manchester's [CSF3](https://ri.itservices.manchester.ac.uk/csf3/) high performance computing system. They each use the same hardware resources: a 4+ core CPU, and an NVidia A100 GPU. There will be a siginificant queue for the A100 if not used with elevated privileges and you may with to swap for a GPU with a shorter queue (e.g. a V100 or A10G).

## Workflow for experiment 1

### **Step 0: [Setup repo/data and copy jobscripts to main folder]**

1. Clone the repo onto CSF.

2. Copy the relevant data files into the `/data/ex1/` and `/data/ex2` folders.

3. Copy the jobscripts kept in this folder into the parent folder (i.e. one level up from this folder) to run.

4. Edit the relevant [settings files](../settings/README.md) for your experiments.

### **Step 1: [Setup Python Environment]**

- **Jobscript**: `jobscript_setup_environment`

- **Python scripts used**: `/tests/test_nps.py`

- **Editable Parameters**: `ENV_NAME`, `PYTHON_VERSION`

- **Notes**: `PYTHON_VERSION` should match the one used by the PyTorch module, so for the default PyTorch 2.3.0 this is python 3.11.

- **stdout filename**: `dppum-setup.oxxxxxxx`

- **How to check it's run correctly**: Make sure there's no errors in the log file. There should then be a section near the end that says:
  
  ```Shell
  Running a test of the environment dppum:
  
  cuda:0
  
  Finished running environment test. It should have said something like 'cuda:0'
  ```
  
  Around 20 lines later there should then be a line saying the test has passed:
  
  ```python
  ========================= 1 passed, 1 warning in 6.06s =========================
  ```

### **Step 2: [Run unit tests]**

- **Jobscript**: `jobscript_unit_tests`
- **Python scripts used**: All `test_*.py` scripts in `/tests/` folder
- **Editable Parameters**: `ENV_NAME`
- **Notes**: ``ENV_NAME`` must match the one from `jobscript_setup_environment`.
- **stdout filename**: `dppum-utests.oxxxxxxx`
- **How to check it's run correctly**: Near the end there should be a similar line to the setup script, saying that a number of tests have passed, and it should not say that any have failed.

### **Step 3: [Train model]**

- **Jobscript**: `jobscript_ex1_train`
- Python scripts used: `/experiments/train.py`
- **Editable Parameters**: `ENV_NAME`, settings file location
- **Notes**: ``ENV_NAME``must match the one from `jobscript_setup_environment`. The default settings file is `settings/ex1/settings_ex1_train.json` near the end of the script. You can add extra lines to run multiple settings files one after the other. You can also retrain an existing model using the `init_weights` key in the settings file.
- **stdout filename**: `ex1-train.oxxxxxxx`
- **How to check it's run correctly**: Check that the output files are created. The word "error" should not appear in the logfile. Near the start there is a similar environment check to the one in environment setup jobscript. You should then see that it has loaded the settings file, metadata for the dataset, and details of the model training. If an error occurs, it is likely to be due to an error in the settings file.
- **Ouptut files**:
  - In the models folder (from the settings file):
    - `train_settings.json`: A copy of the settings file used for training.
    - `training_metrics.csv`: Training metrics (loss and training/validation accuracy vs number of epochs).
    - `weights_epoch_x.pt`: Model weights after `x` epochs of training.
  - In the figures folder (from the settings file):
    - `training_metrics.png`: A plot of the training metrics.

### Step 4: [Evaluate accuracy vs number of training epochs]

- **Jobscript**: `jobscript_ex1_val`
- **Python scripts used**: `/experiments/val.py`
- **Editable Parameters**: `ENV_NAME`, settings file location
- **Notes**: `ENV_NAME`must match the one from `jobscript_setup_environment`. The default settings file is `settings/ex1/settings_ex1_val.json` near the end of the script. You can add extra lines to run multiple settings files one after the other. Make sure you set the models and figures folders to the same as in the training step in your settings file.
- **stdout filename**: `ex1-val.oxxxxxxx`
- **How to check it's run correctly**: Check that the output files are created. The word "error" should not appear in the logfile. Near the start there is a similar environment check to the one in environment setup jobscript.You should then see that it has loaded the settings file, metadata for the dataset, and a note that it is proceeding with the evaluation. If an error occurs, it is likely to be due to an error in the settings file.
- **Ouptut files**:
  - In the models folder (from the settings file):
    - `val_settings.json`: A copy of the settings file used for training.
    - `val_acc_vs_epochs.csv`: Evaluation accuracy vs number of epochs.
  - In the figures folder (from the settings file):
    - `valmetrics.png`: A plot of the evaluation metrics.

### Step 5: [Evaluate accuracy vs number of context trajectories]

- **Jobscript**: `jobscript_ex1_eval_ntraj`
- **Python scripts used**: `/experiments/eval_ntraj.py`
- **Editable Parameters**: `ENV_NAME`, settings file location
- **Notes**: `ENV_NAME`must match the one from `jobscript_setup_environment`. The default settings file is `settings/ex1/settings_ex1_val.json` near the end of the script. You can add extra lines to run multiple settings files one after the other. Make sure you set the models and figures folders to the same as in the training step in your settings file.
- **stdout filename**: `ex1-eval-ntraj.oxxxxxxx`
- **How to check it's run correctly**: Check that the output files are created. The word "error" should not appear in the logfile. Near the start there is a similar environment check to the one in environment setup jobscript. You should then see that it has loaded the settings file, metadata for the dataset, and details of it looping through different numbers of context trajectories. If an error occurs, it is likely to be due to an error in the settings file.
- **Ouptut files**:
  - In the models folder (from the settings file):
    - `eval_ntraj_settings.json`: A copy of the settings file used for training.
    - `eval_acc_vs_ntraj.csv`: Evaluation accuracy vs number of number of context trajectories provided at inference.
  - In the figures folder (from the settings file):
    - `eval_acc_vs_ntraj.png`: A plot of the evaluation metrics.

### Step 6: [Make plots]

The plots created in the above steps are intended to be diagnostic plots only. Users should create their own plots using the data in the CSV files. Some example plottings scripts are given in the [scripts folder](../scripts/).
