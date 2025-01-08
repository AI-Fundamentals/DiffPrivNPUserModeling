# CSF Jobscripts

This folder contains jobscripts designed to run the code on the University of Manchester's [CSF3](https://ri.itservices.manchester.ac.uk/csf3/) high performance computing system. They each use the same hardware resources: a 4+ core CPU, and an NVidia A100 GPU. There will be a siginificant queue for the A100 if not used with elevated privileges and you may with to swap for a GPU with a shorter queue (e.g. a V100 or A10G).

## Workflow

#### **Step 0: [Copy jobscripts to main folder]**

These jobscripts are kept in a folder to keep the code tidy, but they must be copied to the parent folder (i.e. one level up from this folder) to run.

#### **Step 1: [Setup Python Environment]**

- **Jobscript**: `jobscript_setup_environment`

- **Editable Parameters**: `ENV_NAME`, `PYTHON_VERSION`

- **Notes**: `PYTHON_VERSION` should match the one used by the PyTorch module, so for the default PyTorch 2.3.0 this is python 3.11.

- **Logfile name**: `dppum-setup.oxxxxxxx`

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

#### **Step 2: [Run unit tests]**

- **Jobscript**: `jobscript_unit_tests`
- **Editable Parameters**: `ENV_NAME`
- **Notes**: ``ENV_NAME`` must match the one from `jobscript_setup_environment`.
- **Logfile name**: `dppum-utests.oxxxxxxx`
- **How to check it's run correctly**: Near the end there should be a similar line to the setup script, saying that a number of tests have passed, and it should not say that any have failed.

#### **Step 3: [Train model]**

- **Jobscript**: `jobscript_ex1_train` or `jobscript_ex2_train`
- **Editable Parameters**: `ENV_NAME`, settings file location
- **Notes**: ``ENV_NAME``must match the one from `jobscript_setup_environment`. The default settings file is `settings/settings_ex1_train.json` near the end of the script. Also note that you can add extra lines to run multiple settings files one after the other.
- **Logfile name**: `ex1-train.oxxxxxxx`
- **How to check it's run correctly**: Check that the output files are created. The word "error" should not appear in the logfile. Near the start there is a similar environment check to the one in environment setup jobscript. You should then see details of the model training. If an error occurs, it is likely to be due to an error in the settings file.
- **Ouptut files**:
  - In the models folder (from the settings file):
    - `train_settings.json`: A copy of the settings file used for training.
    - `training_metrics.csv`: Training metrics (loss, training and validation accuracy).
    - `weights_epoch_x.pt`: Model weights after `x` epochs of training.
  - In the figures folder (from the settings file):
    - `training_metrics.png`: A plot of the training metrics (not intended for publication).
  
  #### Step 4: [Evaluate accuracy vs number of training epochs]
  
  #### Step 5: [Evaluate accuracy vs number of context trajectories]
  
  #### Step 6: [Make plots]


